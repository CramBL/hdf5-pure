use rstest::rstest;

use super::Allocation;
use super::AllocationKind;
use super::AllocationState;
use super::Context;
use super::Coverage;
use super::Interval;
use super::ObservationError;
use super::PublicationState;
use super::Violation;

#[rstest]
#[case::empty(&[], &[])]
#[case::reordered(&[(8, 4), (0, 4)], &[(0, 4), (8, 4)])]
#[case::adjacent(&[(8, 4), (4, 4)], &[(4, 8)])]
#[case::overlapping(&[(4, 8), (0, 8)], &[(0, 12)])]
#[case::contained(&[(4, 4), (0, 12)], &[(0, 12)])]
#[case::duplicate(&[(4, 4), (4, 4)], &[(4, 4)])]
#[case::empty_member(&[(4, 0), (8, 4)], &[(8, 4)])]
#[case::maximum_end(&[(u64::MAX - 8, 8), (u64::MAX, 0)], &[(u64::MAX - 8, 8)])]
fn coverage_normalizes_interval_unions(
    #[case] input: &[(u64, u64)],
    #[case] expected: &[(u64, u64)],
) {
    assert_eq!(coverage(input).intervals(), intervals(expected));
}

#[rstest]
#[case::difference(SetOperation::Difference)]
#[case::intersection(SetOperation::Intersection)]
fn interval_operations_agree_with_individual_byte_membership(
    #[case] operation: SetOperation,
    #[values(0, 1, 3, 5, 10, 15)] left: u8,
    #[values(0, 1, 3, 5, 10, 15)] right: u8,
) {
    let actual = match operation {
        SetOperation::Difference => bitmap(left).difference(&bitmap(right)),
        SetOperation::Intersection => bitmap(left).intersection(&bitmap(right)),
    };
    let expected = match operation {
        SetOperation::Difference => left & !right,
        SetOperation::Intersection => left & right,
    };
    let observed = (0..4).fold(0, |bits, address| {
        if actual
            .intervals()
            .iter()
            .any(|interval| interval.start() <= address && address < interval.end())
        {
            bits | (1 << address)
        } else {
            bits
        }
    });
    assert_eq!(
        observed, expected,
        "{operation:?}; {left:#06b}, {right:#06b}"
    );
}

#[rstest]
#[case::split_subtraction(
    &[(0, 40)], &[(4, 4), (16, 8)], &[(0, 4), (8, 8), (24, 16)], &[(4, 4), (16, 8)]
)]
#[case::exclusion_spans_multiple_intervals(
    &[(0, 4), (8, 4), (16, 4)], &[(2, 16)], &[(0, 2), (18, 2)], &[(2, 2), (8, 4), (16, 2)]
)]
#[case::touching(
    &[(0, 4)], &[(4, 8)], &[(0, 4)], &[]
)]
#[case::high_addresses(
    &[(u64::MAX - 16, 16)], &[(u64::MAX - 12, 8)],
    &[(u64::MAX - 16, 4), (u64::MAX - 4, 4)], &[(u64::MAX - 12, 8)]
)]
fn interval_operations_preserve_exact_boundaries(
    #[case] left: &[(u64, u64)],
    #[case] right: &[(u64, u64)],
    #[case] remainder: &[(u64, u64)],
    #[case] shared: &[(u64, u64)],
) {
    assert_eq!(
        coverage(left).difference(&coverage(right)),
        coverage(remainder)
    );
    assert_eq!(
        coverage(left).intersection(&coverage(right)),
        coverage(shared)
    );
}

#[rstest]
#[case::before_ownership(AllocationState::Provisional, PublicationState::BeforeOwnership)]
#[case::uncertain(AllocationState::Uncertain, PublicationState::Uncertain)]
fn unpublished_allocations_remain_in_persisted_free_coverage(
    #[case] state: AllocationState,
    #[case] publication: PublicationState,
) {
    super::inspect(
        context(publication),
        &coverage(&[(100, 100)]),
        &coverage(&[(100, 100)]),
        &coverage(&[(0, 20)]),
        &[Allocation::payload(120, 16, state).unwrap()],
    )
    .unwrap();
}

#[rstest]
#[case::linked(AllocationState::Linked, &[(0, 20), (120, 16)])]
#[case::orphaned(AllocationState::Orphaned, &[(0, 20)])]
fn actual_allocations_withdraw_only_their_complete_extent(
    #[case] state: AllocationState,
    #[case] live: &[(u64, u64)],
) {
    super::inspect(
        context(PublicationState::OwnershipPublished),
        &coverage(&[(100, 100)]),
        &coverage(&[(136, 64), (100, 20), (500, 12)]),
        &coverage(live),
        &[Allocation::payload(120, 16, state).unwrap()],
    )
    .unwrap();
}

#[rstest]
#[case::whole_allocation(0, 48)]
#[case::prefix(0, 16)]
#[case::suffix(32, 16)]
fn metadata_allocations_retain_complete_and_parse_ranges(
    #[case] offset: u64,
    #[case] length: u64,
    #[values(MetadataKind::Index, MetadataKind::Manager)] kind: MetadataKind,
) {
    let parsed = interval(120 + offset, length);
    let allocation = kind
        .allocation(interval(120, 48), parsed, AllocationState::Linked)
        .unwrap();
    assert_eq!(allocation.complete(), interval(120, 48));
    assert_eq!(allocation.parsed(), parsed);
    super::inspect(
        context(PublicationState::ReferencesPublished),
        &coverage(&[(100, 100)]),
        &coverage(&[(100, 20), (168, 32)]),
        &coverage(&[(120, 48)]),
        &[allocation],
    )
    .unwrap();
}

#[rstest]
#[case::index(MetadataKind::Index)]
#[case::manager(MetadataKind::Manager)]
fn orphaned_allocations_exclude_their_unparsed_storage_from_free_coverage(
    #[case] kind: MetadataKind,
) {
    let allocation = kind
        .allocation(
            interval(120, 48),
            interval(120, 16),
            AllocationState::Orphaned,
        )
        .unwrap();
    let error = super::inspect(
        context(PublicationState::OwnershipPublished),
        &coverage(&[(100, 100)]),
        &coverage(&[(100, 20), (136, 64)]),
        &Coverage::default(),
        &[allocation],
    )
    .unwrap_err();
    assert_eq!(
        error.violation,
        Violation::OwnedSpaceAdvertisedFree {
            allocation,
            conflicting: coverage(&[(136, 32)])
        }
    );
}

#[test]
fn fresh_eoa_allocations_leave_the_controlled_hole_available() {
    super::inspect(
        context(PublicationState::ReferencesPublished),
        &coverage(&[(100, 100)]),
        &coverage(&[(100, 100)]),
        &coverage(&[(900, 16)]),
        &[Allocation::payload(900, 16, AllocationState::Linked).unwrap()],
    )
    .unwrap();
}

#[rstest]
#[case::provisional(AllocationState::Provisional)]
#[case::uncertain(AllocationState::Uncertain)]
#[case::retired(AllocationState::Retired)]
fn excluded_allocation_states_require_their_unused_hole_coverage(#[case] state: AllocationState) {
    let error = super::inspect(
        context(PublicationState::BeforeOwnership),
        &coverage(&[(100, 100)]),
        &coverage(&[(100, 20), (136, 64)]),
        &Coverage::default(),
        &[Allocation::payload(120, 16, state).unwrap()],
    )
    .unwrap_err();
    assert_eq!(
        error.violation,
        Violation::UnusedSpaceMissing {
            missing: coverage(&[(120, 16)])
        }
    );
}

#[test]
fn retired_storage_can_be_reissued_as_an_actual_allocation() {
    super::inspect(
        context(PublicationState::ReferencesPublished),
        &coverage(&[(100, 100)]),
        &coverage(&[(100, 20), (136, 64)]),
        &coverage(&[(120, 16)]),
        &[
            Allocation::payload(100, 100, AllocationState::Retired).unwrap(),
            Allocation::payload(120, 16, AllocationState::Linked).unwrap(),
        ],
    )
    .unwrap();
}

#[test]
fn empty_controlled_holes_are_rejected() {
    let error = super::inspect(
        context(PublicationState::BeforeOwnership),
        &Coverage::default(),
        &Coverage::default(),
        &Coverage::default(),
        &[],
    )
    .unwrap_err();
    assert_eq!(error.violation, Violation::EmptyControlledHole);
}

#[test]
fn overlapping_active_allocations_are_rejected() {
    let allocations = [
        Allocation::payload(120, 16, AllocationState::Linked).unwrap(),
        Allocation::payload(128, 16, AllocationState::Linked).unwrap(),
    ];
    let error = super::inspect(
        context(PublicationState::ReferencesPublished),
        &coverage(&[(100, 100)]),
        &coverage(&[(100, 20), (144, 56)]),
        &coverage(&[(120, 24)]),
        &allocations,
    )
    .unwrap_err();
    assert_eq!(
        error.violation,
        Violation::OverlappingAllocations {
            allocations: Box::new(allocations),
            conflicting: coverage(&[(128, 8)])
        }
    );
}

#[rstest]
#[case::unused_interval(AllocationState::Linked, &[(120, 16)])]
#[case::orphaned_payload(AllocationState::Orphaned, &[])]
fn missing_unused_space_is_rejected_despite_actual_allocations(
    #[case] state: AllocationState,
    #[case] live: &[(u64, u64)],
) {
    let error = super::inspect(
        context(PublicationState::OwnershipPublished),
        &coverage(&[(100, 100)]),
        &coverage(&[(100, 20)]),
        &coverage(live),
        &[Allocation::payload(120, 16, state).unwrap()],
    )
    .unwrap_err();
    assert_eq!(error.context, context(PublicationState::OwnershipPublished));
    assert_eq!(
        error.violation,
        Violation::UnusedSpaceMissing {
            missing: coverage(&[(136, 64)])
        }
    );
}

#[rstest]
#[case::payload(AllocationKind::Payload)]
#[case::index(AllocationKind::Index)]
#[case::manager(AllocationKind::Manager)]
fn free_coverage_overlapping_live_allocations_is_rejected(#[case] kind: AllocationKind) {
    let allocation = match kind {
        AllocationKind::Payload => Allocation::payload(120, 48, AllocationState::Linked).unwrap(),
        AllocationKind::Index => MetadataKind::Index
            .allocation(
                interval(120, 48),
                interval(120, 16),
                AllocationState::Linked,
            )
            .unwrap(),
        AllocationKind::Manager => MetadataKind::Manager
            .allocation(
                interval(120, 48),
                interval(120, 16),
                AllocationState::Linked,
            )
            .unwrap(),
    };
    let error = super::inspect(
        context(PublicationState::ReferencesPublished),
        &coverage(&[(100, 100)]),
        &coverage(&[(100, 20), (144, 56)]),
        &coverage(&[(120, 48)]),
        &[allocation],
    )
    .unwrap_err();
    assert_eq!(
        error.violation,
        Violation::LiveSpaceAdvertisedFree {
            conflicting: coverage(&[(144, 24)])
        }
    );
}

#[rstest]
#[case::provisional(AllocationState::Provisional)]
#[case::uncertain(AllocationState::Uncertain)]
fn references_before_ownership_exclusion_are_rejected(#[case] state: AllocationState) {
    let allocation = Allocation::payload(120, 16, state).unwrap();
    let error = super::inspect(
        context(PublicationState::BeforeOwnership),
        &coverage(&[(100, 100)]),
        &coverage(&[(100, 100)]),
        &coverage(&[(120, 16)]),
        &[allocation],
    )
    .unwrap_err();
    assert_eq!(
        error.violation,
        Violation::ReferenceBeforeOwnership {
            allocation,
            conflicting: coverage(&[(120, 16)])
        }
    );
}

#[test]
fn referenced_storage_is_rejected_when_reclassified_as_orphaned() {
    let allocation = Allocation::payload(100, 100, AllocationState::Orphaned).unwrap();
    let error = super::inspect(
        context(PublicationState::ReferencesPublished),
        &coverage(&[(100, 100)]),
        &Coverage::default(),
        &coverage(&[(120, 16)]),
        &[allocation],
    )
    .unwrap_err();
    assert_eq!(
        error.violation,
        Violation::ReferencedOrphan {
            allocation,
            conflicting: coverage(&[(120, 16)])
        }
    );
}

#[rstest]
#[case::index(MetadataKind::Index)]
#[case::manager(MetadataKind::Manager)]
fn parse_prefixes_are_rejected_as_complete_reachable_coverage(#[case] kind: MetadataKind) {
    let allocation = kind
        .allocation(
            interval(120, 48),
            interval(120, 16),
            AllocationState::Linked,
        )
        .unwrap();
    let error = super::inspect(
        context(PublicationState::ReferencesPublished),
        &coverage(&[(100, 100)]),
        &coverage(&[(100, 20), (168, 32)]),
        &coverage(&[(120, 16)]),
        &[allocation],
    )
    .unwrap_err();
    assert_eq!(
        error.violation,
        Violation::IncompleteReachableAllocation {
            allocation,
            missing: coverage(&[(136, 32)])
        }
    );
}

#[test]
fn diagnostics_identify_the_case_interruption_state_and_missing_range() {
    let error = super::inspect(
        context(PublicationState::BeforeOwnership),
        &coverage(&[(100, 100)]),
        &coverage(&[(100, 20)]),
        &Coverage::default(),
        &[],
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "case \"controlled hole\", interruption 17, publication BeforeOwnership: UnusedSpaceMissing { missing: Coverage { intervals: [Interval { start: 120, end: 200 }] } }"
    );
}

#[rstest]
#[case::overflow(u64::MAX, 1, ObservationError::RangeOverflow { start: u64::MAX, length: 1 })]
#[case::overflowing_length(4, u64::MAX, ObservationError::RangeOverflow { start: 4, length: u64::MAX })]
fn interval_construction_rejects_overflow(
    #[case] start: u64,
    #[case] length: u64,
    #[case] expected: ObservationError,
) {
    assert_eq!(Interval::new(start, length).unwrap_err(), expected);
}

#[rstest]
#[case::payload(AllocationKind::Payload)]
#[case::index(AllocationKind::Index)]
#[case::manager(AllocationKind::Manager)]
fn allocation_construction_rejects_empty_storage(#[case] kind: AllocationKind) {
    let error = match kind {
        AllocationKind::Payload => Allocation::payload(120, 0, AllocationState::Linked),
        AllocationKind::Index => {
            Allocation::index(interval(120, 0), interval(120, 0), AllocationState::Linked)
        }
        AllocationKind::Manager => {
            Allocation::manager(interval(120, 0), interval(120, 0), AllocationState::Linked)
        }
    }
    .unwrap_err();
    assert_eq!(error, ObservationError::EmptyAllocation { kind });
}

#[rstest]
#[case::index(MetadataKind::Index)]
#[case::manager(MetadataKind::Manager)]
fn metadata_construction_rejects_empty_parse_ranges(#[case] kind: MetadataKind) {
    let error = kind
        .allocation(interval(120, 48), interval(120, 0), AllocationState::Linked)
        .unwrap_err();
    assert_eq!(
        error,
        ObservationError::EmptyAllocation { kind: kind.kind() }
    );
}

#[rstest]
#[case::before_allocation(100, 48)]
#[case::after_allocation(144, 48)]
#[case::surrounding_allocation(100, 100)]
fn allocation_construction_rejects_parse_ranges_outside_complete_storage(
    #[case] start: u64,
    #[case] length: u64,
    #[values(MetadataKind::Index, MetadataKind::Manager)] kind: MetadataKind,
) {
    let complete = interval(120, 48);
    let parsed = interval(start, length);
    let error = kind
        .allocation(complete, parsed, AllocationState::Linked)
        .unwrap_err();
    assert_eq!(
        error,
        ObservationError::ParseOutsideAllocation { complete, parsed }
    );
}

fn context(publication: PublicationState) -> Context<'static> {
    Context {
        case: "controlled hole",
        interruption: 17,
        publication,
    }
}

fn interval(start: u64, length: u64) -> Interval {
    Interval::new(start, length).unwrap()
}

fn intervals(sections: &[(u64, u64)]) -> Vec<Interval> {
    sections
        .iter()
        .map(|&(start, length)| interval(start, length))
        .collect()
}

fn coverage(sections: &[(u64, u64)]) -> Coverage {
    Coverage::new(intervals(sections))
}

fn bitmap(bits: u8) -> Coverage {
    Coverage::new(
        (0..4)
            .filter(|address| bits & (1 << address) != 0)
            .map(|address| interval(address, 1)),
    )
}

#[derive(Clone, Copy, Debug)]
enum SetOperation {
    Difference,
    Intersection,
}

#[derive(Clone, Copy, Debug)]
enum MetadataKind {
    Index,
    Manager,
}

impl MetadataKind {
    fn allocation(
        self,
        complete: Interval,
        parsed: Interval,
        state: AllocationState,
    ) -> Result<Allocation, ObservationError> {
        match self {
            Self::Index => Allocation::index(complete, parsed, state),
            Self::Manager => Allocation::manager(complete, parsed, state),
        }
    }

    fn kind(self) -> AllocationKind {
        match self {
            Self::Index => AllocationKind::Index,
            Self::Manager => AllocationKind::Manager,
        }
    }
}
