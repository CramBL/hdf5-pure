use core::cell::Cell;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::MetadataSource;

use super::{MAX_METADATA_BYTES, ResidualSource};
use crate::error::FormatError;

#[test]
fn residual_group_metadata_preserves_physical_bounds_and_read_budget() {
    let start = PhysicalAddress::new(u64::from(u32::MAX) + 1);
    let range = FileRange::new(start, 4).unwrap();
    let budget = Cell::new(MAX_METADATA_BYTES);
    let source = ResidualSource {
        region: FileRegion::new(range, &[1, 2, 3, 4]).unwrap(),
        remaining: &budget,
    };
    assert_eq!(source.read_metadata(start.get() + 1, 2).unwrap(), [2, 3]);
    assert_eq!(source.read_metadata_at(start.get() + 3, 1).unwrap(), [4]);
    assert_eq!(budget.get(), MAX_METADATA_BYTES - 3);
    let error = source.read_metadata(start.get() - 1, 1).unwrap_err();
    assert_eq!(
        error,
        FormatError::InsufficientRegion {
            required_start: start.get() - 1,
            required_length: 1,
            provided_start: start.get(),
            provided_length: 4
        }
    );
    budget.set(2);
    let error = source.read_metadata(start.get(), 4).unwrap_err();
    assert_eq!(
        error,
        FormatError::ResourceLimit {
            limit: u64::try_from(MAX_METADATA_BYTES).unwrap(),
            required: u64::try_from(MAX_METADATA_BYTES + 2).unwrap()
        }
    );
    assert_eq!(budget.get(), 2);
}

#[test]
fn residual_group_metadata_accepts_the_remaining_budget_before_rejecting_another_read() {
    let start = PhysicalAddress::new(64);
    let budget = Cell::new(MAX_METADATA_BYTES);
    let source = ResidualSource {
        region: FileRegion::new(FileRange::new(start, 4).unwrap(), &[1, 2, 3, 4]).unwrap(),
        remaining: &budget,
    };
    for _ in 0..MAX_METADATA_BYTES / 4 {
        assert_eq!(source.read_metadata(start.get(), 4).unwrap(), [1, 2, 3, 4]);
    }
    assert_eq!(budget.get(), 0);
    assert_eq!(
        source.read_metadata(start.get(), 1).unwrap_err(),
        FormatError::ResourceLimit {
            limit: u64::try_from(MAX_METADATA_BYTES).unwrap(),
            required: u64::try_from(MAX_METADATA_BYTES + 1).unwrap(),
        }
    );
    assert_eq!(budget.get(), 0);
}

use std::sync::atomic::Ordering;

use rstest::rstest;
use test_util::group_info::compact_messages;
use test_util::object_header::v2::Header;
use test_util::object_header::{Message, MessageType, v1};
use test_util_hdf5::global_heap_fixture::{Fixture, Form, hard_link};

use super::{MAX_RESIDUAL_BYTES, MAX_RESIDUAL_GROUPS};
use crate::edit::analysis::AnalysisPhase;
use crate::edit::test_util as edit_test_util;
use crate::object_graph::ObjectId;

#[rstest]
#[case::descendants(Shape::Groups(1), true)]
#[case::group_limit(Shape::Groups(MAX_RESIDUAL_GROUPS), true)]
#[case::too_many_groups(Shape::Groups(MAX_RESIDUAL_GROUPS + 1), false)]
#[case::byte_limit(Shape::Padded(MAX_RESIDUAL_BYTES), true)]
#[case::too_many_bytes(Shape::Padded(MAX_RESIDUAL_BYTES + 1), false)]
#[case::unknown_message(Shape::Unknown, false)]
#[case::malformed_header(Shape::Malformed, false)]
#[case::legacy_header(Shape::Legacy, false)]
#[case::residual_target(Shape::SelfLink, false)]
#[case::unresolved_target(Shape::Unresolved, false)]
fn residual_groups_preserve_original_descendants_within_limits(
    #[case] shape: Shape,
    #[case] admitted: bool,
) {
    let mut fixture = Fixture::new();
    let records =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &37_i32.to_le_bytes());
    let leaf = fixture.append_header(records);
    let child = fixture.append_header(compact_messages(&[hard_link("leaf", leaf)]));
    let branch = fixture.append_header(compact_messages(&[hard_link("child", child)]));
    fixture.link("branch", branch);
    let residual = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let records = |target| compact_messages(&[hard_link("branch", target)]);
    let encoded = |target| Header::new().messages(records(target)).build();
    let bytes = match shape {
        Shape::Groups(count) => encoded(branch).repeat(count),
        Shape::Padded(length) => {
            let mut bytes = encoded(branch);
            bytes.resize(length, 0);
            bytes
        }
        Shape::Unknown => Header::new()
            .messages(compact_messages(&[Message::new(MessageType(0x7f), &[1])]))
            .build(),
        Shape::Malformed => b"OHDR\x02".to_vec(),
        Shape::Legacy => v1::Header::new().messages(records(branch)).build(),
        Shape::SelfLink => encoded(residual),
        Shape::Unresolved => encoded(residual + 1),
    };
    fixture.image.append(&bytes);
    let path = test_util::temp::temp_path("residual-compact-group.h5");
    fixture.write(&path);
    let (engine, reads) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let id = |address| ObjectId::new(PhysicalAddress::new(address));
    let source = phase
        .reconstruction_source(&engine, id(Fixture::ROOT))
        .unwrap();
    let result = phase.residual_group_domain(&source).unwrap();
    assert_eq!(result.is_some(), admitted);
    if let Some(groups) = result.as_ref() {
        assert!(groups.roots().contains(&id(residual)));
        assert_eq!(
            groups
                .objects()
                .difference(groups.roots())
                .copied()
                .collect::<Vec<_>>(),
            [id(leaf), id(child), id(branch)]
        );
        assert_eq!(
            groups.roots().len(),
            match shape {
                Shape::Groups(count) => count,
                _ => 1,
            }
        );
    }
    let before = reads.load(Ordering::Relaxed);
    assert_eq!(
        phase.residual_group_domain(&source).unwrap().is_some(),
        admitted
    );
    assert_eq!(reads.load(Ordering::Relaxed), before);
}

enum Shape {
    Groups(usize),
    Padded(usize),
    Unknown,
    Malformed,
    Legacy,
    SelfLink,
    Unresolved,
}
