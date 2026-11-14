use std::fs;
use std::path::Path;
use std::sync::Arc;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_space::__private::{FreeClass, PageType};
use rstest::rstest;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture;

use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationPublication;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::commit::CommitOutcome;
use crate::edit::commit::PublicationState;
use crate::edit::commit::plan::inventory::members::ParentDisposition;
use crate::edit::commit::plan::inventory::prospective::ReferencePresence;
use crate::edit::commit::plan::inventory::{InventoryPreparation, PreparedEditInventory};
use crate::image::WriteBuffering;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;
use crate::type_builders::DatasetBuilder;
use crate::{
    AttrValue, EditStorageConflict, Error, File, FileBuilder, FileSpaceSettings, FileSpaceStrategy,
    PhysicalAddress,
};

#[rstest]
#[case::flat_unbuffered(false, false)]
#[case::flat_session(false, true)]
#[case::page_unbuffered(true, false)]
#[case::page_session(true, true)]
fn session_parent_retirement_preserves_unrelated_unknown_storage(
    #[case] page: bool,
    #[case] buffered: bool,
    #[values(false, true)] mirror: bool,
) {
    let path = test_util::temp::temp_path("session-parent-retirement.h5");
    let (mut engine, other) = fixture(&path, page, mirror);
    let original = engine.disk_bytes().unwrap();
    let parent = engine.vl_overwrite_heaps[&ObjectPathBuf::parse("labels")][0];
    stage(&mut engine);
    let snapshot = engine.snapshot_free();
    let prepared = prepare(&mut engine);
    assert!(prepared.original.owners().is_none());
    assert_eq!(prepared.dispositions.session_parents.len(), 1);
    let witness = &prepared.dispositions.session_parents[&parent.start()];
    assert_eq!(witness.range, parent);
    let class = if page {
        FreeClass::Dead
    } else {
        FreeClass::Page(PageType::Raw)
    };
    assert_eq!(witness.class, class);
    engine
        .set_write_buffering(if buffered {
            WriteBuffering::Session {
                page_size: 4096,
                max_bytes: 16_384,
            }
        } else {
            WriteBuffering::Unbuffered
        })
        .unwrap();
    let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
    assert_eq!(
        pending.prospective().parent_presence(parent.start()),
        ReferencePresence::Unavailable
    );
    let decisions = pending.member_decisions();
    let ParentDisposition::Retire(retirement) = &decisions.dispositions()[&parent.start()] else {
        panic!("expected session parent retirement");
    };
    assert_eq!(retirement.range(), parent);
    assert_eq!(retirement.class(), class);
    assert_eq!(decisions.plans().count(), 0);
    assert!(matches!(
        decisions.dispositions()[&other.start()],
        ParentDisposition::Retain
    ));
    pending.exclusion().unwrap();
    let completion = pending.publish_compacting_members().ok().unwrap();
    completion.mutation.as_ref().unwrap();
    let ParentDisposition::Retire(retirement) =
        &completion.decisions.dispositions()[&parent.start()]
    else {
        panic!("expected consumed session parent retirement");
    };
    assert_eq!(retirement.range(), parent);
    assert_eq!(retirement.class(), class);
    engine
        .finish_commit(snapshot, completion.into_outcome())
        .unwrap();
    assert!(
        engine.image().len() <= parent.start().get()
            || engine
                .reusable_ranges()
                .any(|range| range.unwrap().contains(parent))
    );
    drop(engine);
    let after = fs::read(&path).unwrap();
    let other =
        usize::try_from(other.start().get()).unwrap()..usize::try_from(other.end().get()).unwrap();
    assert_eq!(&after[other.clone()], &original[other]);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("labels").unwrap().read_string().unwrap(),
        ["second"]
    );
    assert_eq!(file.root().attrs().unwrap()["changed"], AttrValue::I64(19));
    assert_eq!(
        file.dataset("unknown").unwrap().read_raw().unwrap(),
        null_reference()
    );
}

#[rstest]
#[case::session(Restriction::Session)]
#[case::source_phase(Restriction::Source)]
#[case::owner(Restriction::Owner)]
fn session_parent_retirement_requires_matching_associations(#[case] restriction: Restriction) {
    let path = test_util::temp::temp_path("session-parent-association.h5");
    let (mut engine, _) = fixture(&path, false, false);
    let parent = engine.vl_overwrite_heaps[&ObjectPathBuf::parse("labels")][0];
    stage(&mut engine);
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let other_source = AnalysisPhase::new(&engine)
        .reconstruction_source(&engine, root)
        .unwrap();
    let mut prepared = prepare(&mut engine);
    let witness = prepared
        .dispositions
        .session_parents
        .get_mut(&parent.start())
        .unwrap();
    match restriction {
        Restriction::Owner => witness.owner = ObjectId::new(parent.start()),
        Restriction::Session => witness.session = AllocationPublication::new().session(),
        Restriction::Source => witness.source = other_source,
    }
    let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
    assert!(matches!(
        pending.member_decisions().dispositions()[&parent.start()],
        ParentDisposition::Retain
    ));
}

#[rstest]
#[case::copy(Aliasing::Copy)]
#[case::raw_write(Aliasing::RawWrite)]
fn session_parent_capture_withholds_retirement_for_reference_sharing(#[case] aliasing: Aliasing) {
    let path = test_util::temp::temp_path("session-parent-aliasing.h5");
    let (mut engine, _) = fixture(&path, false, false);
    stage(&mut engine);
    match aliasing {
        Aliasing::Copy => engine
            .copy(
                &ObjectPathBuf::parse("unknown"),
                &ObjectPathBuf::parse("copy"),
            )
            .unwrap(),
        Aliasing::RawWrite => {
            let mut values = DatasetBuilder::new("");
            values.with_raw_data(global_heap_fixture::object_reference(), null_reference(), 1);
            engine
                .stage_dataset_write(&ObjectPathBuf::parse("unknown"), values)
                .unwrap();
        }
    }
    let prepared = prepare(&mut engine);
    assert!(prepared.dispositions.session_parents.is_empty());
}

#[rstest]
#[case::unobserved_member(CaptureRestriction::Unobserved)]
#[case::range(CaptureRestriction::Range)]
fn session_parent_capture_requires_complete_original_records(
    #[case] restriction: CaptureRestriction,
) {
    let path = test_util::temp::temp_path("session-parent-original.h5");
    let (mut engine, _) = fixture(&path, false, false);
    let dataset_path = ObjectPathBuf::parse("labels");
    let parent = engine.vl_overwrite_heaps[&dataset_path][0];
    match restriction {
        CaptureRestriction::Range => {
            engine.vl_overwrite_heaps.insert(
                dataset_path,
                vec![FileRange::new(parent.start(), parent.length() - 8).unwrap()],
            );
        }
        CaptureRestriction::Unobserved => {
            let bytes = global_heap::collection(
                &[
                    global_heap::Object::new(1, b"first"),
                    global_heap::Object::new(99, b"unobserved"),
                ],
                Widths::EIGHT,
            );
            assert_eq!(u64::try_from(bytes.len()).unwrap(), parent.length());
            engine.image.write_at(parent.start().get(), &bytes).unwrap();
            engine.image.flush().unwrap();
        }
    }
    stage(&mut engine);
    let prepared = prepare(&mut engine);
    assert!(prepared.dispositions.session_parents.is_empty());
    let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
    assert!(matches!(
        pending.member_decisions().dispositions()[&parent.start()],
        ParentDisposition::Retain
    ));
}

#[test]
fn session_parent_retirement_preserves_inherited_retirement_conflicts() {
    let path = test_util::temp::temp_path("session-parent-conflict.h5");
    let (mut engine, _) = fixture(&path, false, false);
    let parent = engine.vl_overwrite_heaps[&ObjectPathBuf::parse("labels")][0];
    stage(&mut engine);
    let prepared = prepare(&mut engine);
    let conflict = FileRange::new(parent.start().checked_add(8).unwrap(), 8).unwrap();
    engine.superseded_heaps.push(conflict);
    let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
    let failure = match pending.publish_compacting_members() {
        Ok(_) => panic!("expected conflicting parent retirement"),
        Err(failure) => failure,
    };
    let outcome = (*failure).into_outcome();
    let CommitOutcome::Applied(result) = outcome else {
        panic!("expected consumed pre-publication failure");
    };
    let error = result.unwrap_err();
    let Error::EditStorageConflict(EditStorageConflict::Overlap { first, second }) = error else {
        panic!("unexpected retirement error: {error:?}");
    };
    assert_eq!((first.start(), first.end()), (parent.start(), parent.end()));
    assert_eq!(
        (second.start(), second.end()),
        (conflict.start(), conflict.end())
    );
    assert!(!engine.publication.attempted());
}

fn fixture(path: &Path, page: bool, mirror: bool) -> (WriteEngine, FileRange) {
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(if page {
            FileSpaceStrategy::Page
        } else {
            FileSpaceStrategy::FsmAggr
        })
        .with_persistence(true),
    );
    builder.set_attr("changed", AttrValue::VarLenAsciiString("before".into()));
    builder
        .create_dataset("labels")
        .with_vlen_strings(&["seed"]);
    builder.create_dataset("unknown").with_raw_data(
        global_heap_fixture::object_reference(),
        null_reference(),
        1,
    );
    builder.write(path).unwrap();
    let bytes = fs::read(path).unwrap();
    let ranges = global_heap::collection_ranges(&bytes, Widths::EIGHT);
    assert_eq!(ranges.len(), 2);
    let mut engine = if mirror {
        WriteEngine::open_source_only(path).unwrap()
    } else {
        WriteEngine::open_bounded_counting(path, Arc::default()).unwrap()
    };
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, root).unwrap();
    let references = phase.heap_reference_facts(&source).unwrap();
    assert_eq!(references.collections().len(), 1);
    let address = *references.collections().iter().next().unwrap();
    let range = ranges.iter().find(|range| range.start == address).unwrap();
    let other = FileRange::new(PhysicalAddress::new(range.start), range.end - range.start).unwrap();
    drop(phase);
    let mut values = DatasetBuilder::new("");
    values.with_vlen_strings(&["first"]);
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("labels"), values)
        .unwrap();
    engine.commit().unwrap();
    assert!(!engine.global_heap_liveness().unwrap().analysis_complete);
    (engine, other)
}

fn stage(engine: &mut WriteEngine) {
    let mut values = DatasetBuilder::new("");
    values.with_vlen_strings(&["second"]);
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("labels"), values)
        .unwrap();
    engine
        .set_group_attr(&ObjectPathBuf::root(), "changed", AttrValue::I64(19))
        .unwrap();
}

fn null_reference() -> Vec<u8> {
    vec![0; usize::try_from(global_heap_fixture::object_reference().type_size()).unwrap()]
}

fn prepare(engine: &mut WriteEngine) -> PreparedEditInventory {
    engine.publication = PublicationState::Ready;
    let staged = std::mem::take(&mut engine.staged);
    match PreparedEditInventory::prepare_observed(engine, staged) {
        InventoryPreparation::Tree(prepared) => prepared,
        InventoryPreparation::Rejected { error, .. } => panic!("preparation failed: {error:?}"),
        InventoryPreparation::Direct(_) => panic!("expected checked tree preparation"),
    }
}

#[derive(Clone, Copy, Debug)]
enum Restriction {
    Owner,
    Session,
    Source,
}

#[derive(Clone, Copy, Debug)]
enum Aliasing {
    Copy,
    RawWrite,
}

#[derive(Clone, Copy, Debug)]
enum CaptureRestriction {
    Range,
    Unobserved,
}

test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}
