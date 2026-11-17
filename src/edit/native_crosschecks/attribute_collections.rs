use rstest::rstest;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::global_heap::attributes as fixture;
use test_util_hdf5::global_heap::attributes::{Edit, Owner, Storage};

use crate::{AttrValue, File};

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
    let file = File::open_rw(&path).unwrap();
    let value = match edit {
        Edit::Numeric => Some(AttrValue::I64(19)),
        Edit::Remove => None,
        Edit::Variable => Some(AttrValue::VarLenAsciiString("after".into())),
    };
    match owner {
        Owner::Dataset => {
            let mut dataset = file.dataset(owner.path()).unwrap();
            match value {
                Some(value) => dataset.set_attr("label", value).unwrap(),
                None => dataset.remove_attr("label").unwrap(),
            }
        }
        Owner::Group | Owner::Root => {
            let group = if matches!(owner, Owner::Root) {
                file.root()
            } else {
                file.group(owner.path()).unwrap()
            };
            match value {
                Some(value) => group.set_attr("label", value).unwrap(),
                None => group.remove_attr("label").unwrap(),
            }
        }
    }
    file.commit().unwrap();
    let free = file.reusable_free_sections().unwrap();
    assert!(
        free.sections().iter().any(|section| {
            section.address().get() <= collection.start
                && collection.end <= section.end_address().get()
        }),
        "original collection {collection:?} is not reusable in {free:?}"
    );
    file.close().unwrap();
    fixture::assert_values(&path, owner, storage, edit);
}
