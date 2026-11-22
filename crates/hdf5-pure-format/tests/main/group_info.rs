use hdf5_pure_format::__private::{
    GroupEstimates, GroupInfo, GroupInfoOutputTooSmall, GroupPhaseChange,
};
use hdf5_pure_format::FormatError;
use rstest::rstest;
use test_util::group_info::{self, Estimates, PhaseChange};
use test_util::object_header::v1::MESSAGE_ALIGNMENT;

#[rstest]
#[case::omitted(None)]
#[case::explicit_defaults(Some((8, 6)))]
#[case::always_dense(Some((0, 0)))]
#[case::equal(Some((1, 1)))]
#[case::one_way(Some((12, 0)))]
#[case::maximum(Some((u16::MAX, u16::MAX)))]
#[case::high_bytes(Some((0x1234, 0x0567)))]
fn parses_stored_fields_and_effective_thresholds(
    #[case] phase_change: Option<(u16, u16)>,
    #[values(None, Some((4, 8)), Some((0, 0)), Some((0x1234, 0xabcd)))] estimates: Option<(
        u16,
        u16,
    )>,
) {
    let fixture = fixture(phase_change, estimates);
    let (stored, estimates) = fields(fixture);
    let info = GroupInfo::parse(&fixture.build()).unwrap();
    assert_eq!(info, GroupInfo::new(stored, estimates));
    assert_eq!(info.stored_phase_change(), stored);
    assert_eq!(info.estimates(), estimates);
    assert_eq!(
        (
            info.phase_change().max_compact(),
            info.phase_change().min_dense()
        ),
        phase_change.unwrap_or((8, 6)),
    );
}

#[rstest]
#[case::neither(None, None)]
#[case::phase_change(Some((8, 6)), None)]
#[case::estimates(None, Some((4, 8)))]
#[case::both(Some((8, 6)), Some((4, 8)))]
#[case::always_dense(Some((0, 0)), Some((0, 0)))]
#[case::equal(Some((1, 1)), None)]
#[case::maximum(Some((u16::MAX, u16::MAX)), Some((u16::MAX, u16::MAX)))]
#[case::high_bytes(Some((0x1234, 0x0567)), Some((0x1357, 0x2468)))]
fn bounded_output_matches_independent_bytes(
    #[case] phase_change: Option<(u16, u16)>,
    #[case] estimates: Option<(u16, u16)>,
    #[values(0, 4)] suffix: usize,
) {
    let fixture = fixture(phase_change, estimates);
    let (stored, estimates) = fields(fixture);
    let info = GroupInfo::new(stored, estimates);
    let body = fixture.build();
    let mut output = vec![0xa5; 1 + body.len() + suffix];
    assert_eq!(info.encoded_size(), body.len());
    assert_eq!(info.write_into(&mut output[1..]).unwrap(), body.len());
    let mut expected = vec![0xa5; output.len()];
    expected[1..1 + body.len()].copy_from_slice(&body);
    assert_eq!(output, expected);
    assert_eq!(info.serialize(), body);
}

#[rstest]
#[case::neither(None, None)]
#[case::phase_change(Some((8, 6)), None)]
#[case::estimates(None, Some((4, 8)))]
#[case::both(Some((8, 6)), Some((4, 8)))]
fn short_output_preserves_bytes(
    #[case] phase_change: Option<(u16, u16)>,
    #[case] estimates: Option<(u16, u16)>,
) {
    let fixture = fixture(phase_change, estimates);
    let (stored, estimates) = fields(fixture);
    let info = GroupInfo::new(stored, estimates);
    let required = fixture.build().len();
    for available in 0..required {
        let mut output = [0xa5; 12];
        assert_eq!(
            info.write_into(&mut output[1..1 + available]).unwrap_err(),
            GroupInfoOutputTooSmall {
                required,
                available
            },
        );
        assert_eq!(output, [0xa5; 12]);
    }
}

#[rstest]
#[case::empty(None, None, 0, 2)]
#[case::missing_flags(None, None, 1, 2)]
#[case::phase_change(Some((8, 6)), None, 5, 6)]
#[case::estimates(None, Some((4, 8)), 5, 6)]
#[case::first_pair(Some((8, 6)), Some((4, 8)), 5, 10)]
#[case::last_pair(Some((8, 6)), Some((4, 8)), 9, 10)]
fn reports_the_required_extent_of_a_truncation(
    #[case] phase_change: Option<(u16, u16)>,
    #[case] estimates: Option<(u16, u16)>,
    #[case] actual: usize,
    #[case] expected: usize,
) {
    let bytes = fixture(phase_change, estimates).build();
    let body = &bytes[..actual];
    let error = FormatError::InvalidGroupInfoLength { expected, actual };
    assert_eq!(GroupInfo::unpadded_body_len(body).unwrap_err(), error);
    assert_eq!(GroupInfo::parse(body).unwrap_err(), error);
}

#[rstest]
#[case::version_without_flags(&[1], FormatError::InvalidGroupInfoLength { expected: 2, actual: 1 })]
#[case::version_with_reserved_flags(&[1, 0x83], FormatError::InvalidGroupInfoVersion(1))]
#[case::reserved_flags_without_payload(&[0, 0x83], FormatError::InvalidGroupInfoFlags(0x83))]
fn checks_prefix_length_version_and_flags_before_body(
    #[case] bytes: &[u8],
    #[case] error: FormatError,
) {
    assert_eq!(GroupInfo::unpadded_body_len(bytes).unwrap_err(), error);
    assert_eq!(GroupInfo::parse(bytes).unwrap_err(), error);
}

#[rstest]
#[case::neither(None, None, 2)]
#[case::phase_change(Some((8, 6)), None, 6)]
#[case::estimates(None, Some((4, 8)), 6)]
#[case::both(Some((8, 6)), Some((4, 8)), 10)]
fn accepts_padding_for_prefix_reads_and_rejects_it_for_exact_reads(
    #[case] phase_change: Option<(u16, u16)>,
    #[case] estimates: Option<(u16, u16)>,
    #[case] expected: usize,
) {
    let fixture = fixture(phase_change, estimates);
    let (stored, estimates) = fields(fixture);
    let mut body = fixture.build();
    body.resize(body.len().next_multiple_of(MESSAGE_ALIGNMENT), 0);
    assert_eq!(GroupInfo::unpadded_body_len(&body).unwrap(), expected);
    assert_eq!(
        GroupInfo::parse_padded(&body).unwrap(),
        GroupInfo::new(stored, estimates)
    );
    assert_eq!(
        GroupInfo::parse(&body).unwrap_err(),
        FormatError::InvalidGroupInfoLength {
            expected,
            actual: body.len()
        },
    );
}

#[rstest]
#[case::truncated_estimates(9)]
#[case::trailing_byte(11)]
fn length_errors_precede_inverted_thresholds(#[case] actual: usize) {
    let mut body = fixture(Some((8, 6)), Some((4, 8))).build();
    body[4] = 9;
    body.resize(actual, 0);
    assert_eq!(
        GroupInfo::parse(&body).unwrap_err(),
        FormatError::InvalidGroupInfoLength {
            expected: 10,
            actual
        },
    );
}

#[test]
fn prefix_discovery_leaves_threshold_ordering_to_decoding() {
    let mut body = fixture(Some((8, 6)), None).build();
    body[4] = 9;
    assert_eq!(GroupInfo::unpadded_body_len(&body).unwrap(), 6);
    let error = FormatError::InvalidGroupInfoThresholds {
        max_compact: 8,
        min_dense: 9,
    };
    assert_eq!(GroupInfo::parse(&body).unwrap_err(), error);
    body.extend_from_slice(&[0, 0]);
    assert_eq!(GroupInfo::parse_padded(&body).unwrap_err(), error);
}

#[rstest]
#[case::inverted(3, 4)]
#[case::zero_maximum(0, 1)]
fn rejects_inverted_thresholds_at_construction(#[case] max_compact: u16, #[case] min_dense: u16) {
    assert_eq!(
        GroupPhaseChange::new(max_compact, min_dense).unwrap_err(),
        FormatError::InvalidGroupInfoThresholds {
            max_compact,
            min_dense
        },
    );
}

fn fixture(
    phase_change: Option<(u16, u16)>,
    estimates: Option<(u16, u16)>,
) -> group_info::GroupInfo {
    group_info::GroupInfo {
        phase_change: phase_change.map(|(max, min)| PhaseChange::new(max, min).unwrap()),
        estimates: estimates.map(|(entries, name_length)| Estimates {
            entries,
            name_length,
        }),
    }
}

fn fields(fixture: group_info::GroupInfo) -> (Option<GroupPhaseChange>, Option<GroupEstimates>) {
    (
        fixture.phase_change.map(|phase| {
            let (max, min) = phase.thresholds();
            GroupPhaseChange::new(max, min).unwrap()
        }),
        fixture.estimates.map(
            |Estimates {
                 entries,
                 name_length,
             }| GroupEstimates {
                entries,
                name_length,
            },
        ),
    )
}
