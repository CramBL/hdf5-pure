use rstest::rstest;
use test_util::global_heap;
use test_util::snapshot::text::{Report, Value};
use test_util::widths::Widths;
use test_util_hdf5::global_heap::attributes as fixture;
use test_util_hdf5::global_heap::attributes::Edit;
use test_util_hdf5::global_heap::attributes::Owner;
use test_util_hdf5::global_heap::attributes::Storage;

use crate::AttrValue;
use crate::PhysicalAddress;
use crate::access_mode::AccessMode;
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::heap::associations::MemberPresence;
use crate::edit::analysis::physical::claims::coverage::content::ResidualOutcome;
use crate::edit::attributes;
use crate::edit::test_util as edit_test_util;
use crate::group::v2 as group_v2;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::compact(Storage::Compact)]
#[case::dense(Storage::Dense)]
fn native_attribute_edits_retire_complete_original_collections(
    #[case] storage: Storage,
    #[values(Owner::Dataset, Owner::Group, Owner::Root)] owner: Owner,
    #[values(Edit::Numeric, Edit::Remove, Edit::Variable)] edit: Edit,
) {
    let path = test_util::temp::temp_path("native-attribute-collection.h5");
    fixture::write(&path, owner, storage);
    let original = std::fs::read(&path).unwrap();
    let ranges = global_heap::collection_ranges(&original, Widths::EIGHT);
    let [collection] = ranges.as_slice() else {
        panic!("expected one original collection: {ranges:?}");
    };
    assert!(collection.end < u64::try_from(original.len()).unwrap());
    let mut engine = edit_test_util::open_engine(&path);
    let owner_path = ObjectPathBuf::parse(owner.path());
    let address = group_v2::resolve_path_any_from_source(
        &engine.opened_source(),
        AccessMode::ReadWrite,
        engine.context.superblock(),
        engine.context.widths(),
        &owner_path.as_path(),
    )
    .unwrap();
    let region = WriteEngine::gather_oh_messages(
        engine.image(),
        PhysicalAddress::new(address),
        engine.context.addresses(),
        engine.context.widths(),
    )
    .unwrap();
    assert_eq!(
        attributes::find_attribute_info(&region)
            .unwrap()
            .and_then(|(_, _, info)| info.fractal_heap_address)
            .is_some(),
        matches!(storage, Storage::Dense)
    );
    let mut phase = AnalysisPhase::new(&engine);
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let source = phase.reconstruction_source(&engine, root).unwrap();
    let associations = phase.member_associations(&source).unwrap();
    assert!(associations.topology_available());
    assert!(associations.incomplete_references().is_empty());
    assert_eq!(associations.collections().len(), 1);
    let association = &associations.collections()[&PhysicalAddress::new(collection.start)];
    assert!(association.error().is_none());
    assert!(association.unmatched().is_empty());
    assert_eq!(association.observed().len(), 1);
    let observed = association.observed().values().next().unwrap();
    assert_eq!(observed.presence(), MemberPresence::Allocated);
    assert_eq!(
        observed.referrers(),
        [ObjectId::new(PhysicalAddress::new(address))]
    );
    let claims = phase.physical_claims(&source).unwrap();
    assert!(claims.problems().is_empty(), "{:?}", claims.problems());
    let coverage = phase.physical_coverage(&source).unwrap();
    assert_eq!(coverage.conflict(), None);
    assert!(coverage.beyond_source().is_empty());
    assert!(coverage.beyond_eoa().is_empty());
    assert!(!coverage.residuals().is_empty());
    let expected_residuals = coverage.residuals().to_vec();
    let content = phase.residual_content(&source).unwrap();
    assert_eq!(
        content
            .intervals()
            .iter()
            .map(|entry| entry.range())
            .collect::<Vec<_>>(),
        expected_residuals
    );
    let mut report = Report::new();
    report.table(
        "Residual content",
        [
            "Start",
            "Length",
            "Zero prefix",
            "Unexamined",
            "Outcome",
            "Read calls",
            "Requested bytes",
        ],
        content.intervals().iter().map(|entry| {
            let outcome = match entry.outcome() {
                ResidualOutcome::AllZero => "all zero",
                ResidualOutcome::NonZero { .. } => "nonzero",
                ResidualOutcome::ReadFailed(error) => {
                    panic!("native residual read failed: {error:?}")
                }
            };
            assert_eq!(
                entry.zero_prefix().length()
                    + entry.unexamined().length()
                    + u64::from(matches!(entry.outcome(), ResidualOutcome::NonZero { .. })),
                entry.range().length()
            );
            [
                Value::Number(entry.range().start().get()),
                Value::Number(entry.range().length()),
                Value::Number(entry.zero_prefix().length()),
                Value::Number(entry.unexamined().length()),
                Value::text(outcome),
                Value::Number(entry.read_calls()),
                Value::Number(entry.requested_bytes()),
            ]
        }),
    );
    eprintln!("{owner:?}/{storage:?}/{edit:?}\n{}", report.finish());
    drop(phase);
    let value = match edit {
        Edit::Numeric => Some(AttrValue::I64(19)),
        Edit::Remove => None,
        Edit::Variable => Some(AttrValue::VarLenAsciiString("after".into())),
    };
    match (owner, value) {
        (Owner::Dataset, Some(value)) => engine
            .set_dataset_attr(&owner_path, "label", value)
            .unwrap(),
        (Owner::Dataset, None) => engine.remove_dataset_attr(&owner_path, "label").unwrap(),
        (Owner::Group | Owner::Root, Some(value)) => {
            engine.set_group_attr(&owner_path, "label", value).unwrap()
        }
        (Owner::Group | Owner::Root, None) => {
            engine.remove_group_attr(&owner_path, "label").unwrap()
        }
    }
    engine.commit().unwrap();
    let free = engine.reusable_free_sections().unwrap();
    assert!(
        free.sections().iter().any(|section| {
            section.address().get() <= collection.start
                && collection.end <= section.end_address().get()
        }),
        "original collection {collection:?} is not reusable in {free:?}"
    );
    let liveness = engine.global_heap_liveness().unwrap();
    assert!(liveness.analysis_complete);
    assert!(!liveness.referenced_collections.contains(&collection.start));
    assert_eq!(
        liveness.referenced_collections.len(),
        usize::from(matches!(edit, Edit::Variable))
    );
    drop(engine);
    fixture::assert_values(&path, owner, storage, edit);
}
