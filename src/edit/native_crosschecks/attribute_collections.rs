use rstest::rstest;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::global_heap::attributes as fixture;
use test_util_hdf5::global_heap::attributes::Edit;
use test_util_hdf5::global_heap::attributes::Owner;
use test_util_hdf5::global_heap::attributes::Storage;

use crate::AttrValue;
use crate::access_mode::AccessMode;
use crate::edit::WriteEngine;
use crate::edit::attributes;
use crate::edit::test_util as edit_test_util;
use crate::group::v2 as group_v2;
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
        address,
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
