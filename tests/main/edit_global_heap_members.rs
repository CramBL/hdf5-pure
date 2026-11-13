use std::fs;
use std::ops::Range;

use hdf5_pure::AttrValue;
use hdf5_pure::CharacterSet;
use hdf5_pure::Datatype;
use hdf5_pure::File;
use hdf5_pure::StringPadding;
use hdf5_pure::VariableLengthType;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
use hdf5_pure_format::LengthWidth;
use rstest::rstest;
use test_util::global_heap::MIN_COLLECTION_SIZE;
use test_util::global_heap::Object;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

#[rstest]
#[case::attribute_removal(Edit::RemoveAttribute)]
#[case::attribute_replacement(Edit::ReplaceAttribute)]
#[case::dataset_overwrite(Edit::Overwrite)]
#[case::dataset_deletion(Edit::Delete)]
fn public_commit_reclaims_obsolete_members_and_keeps_shared_values(#[case] edit: Edit) {
    let path = test_util::temp::temp_path("public-heap-members.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[
        Object {
            index: 7,
            reference_count: 0,
            data: b"removed",
        },
        Object {
            index: 91,
            reference_count: 3,
            data: b"keep",
        },
    ]);
    let datatype = Datatype::VariableLength(VariableLengthType::string(
        StringPadding::NullTerminate,
        CharacterSet::Utf8,
    ));
    let reference = global_heap_fixture::reference(7, parent, 7);
    let records = match edit {
        Edit::RemoveAttribute | Edit::ReplaceAttribute => {
            let mut records = Form::CompactDataset.messages(
                &mut fixture,
                &hdf5_pure::make_i32_type(),
                &17_i32.to_le_bytes(),
            );
            records.extend(Form::CompactAttribute.messages(&mut fixture, &datatype, &reference));
            records
        }
        Edit::Overwrite | Edit::Delete => {
            Form::ContiguousDataset.messages(&mut fixture, &datatype, &reference)
        }
    };
    fixture.object("value", records);
    let records = Form::ContiguousDataset.messages(
        &mut fixture,
        &datatype,
        &global_heap_fixture::reference(4, parent, 91),
    );
    fixture.object("keep", records);
    fixture.write(&path);
    let before = fs::read(&path).unwrap();
    assert_eq!(
        members(&before, parent),
        [(7, 0, b"removed".to_vec()), (91, 3, b"keep".to_vec())]
    );
    let file = File::open_rw(&path).unwrap();
    match edit {
        Edit::RemoveAttribute => file.dataset("value").unwrap().remove_attr("a").unwrap(),
        Edit::ReplaceAttribute => file
            .dataset("value")
            .unwrap()
            .set_attr("a", AttrValue::I64(9))
            .unwrap(),
        Edit::Overwrite => file
            .dataset("value")
            .unwrap()
            .write_staged(|dataset| {
                dataset.with_vlen_strings(&["after"]);
            })
            .unwrap(),
        Edit::Delete => file.root().delete("value").unwrap(),
    }
    file.commit().unwrap();
    file.close().unwrap();
    let after = fs::read(&path).unwrap();
    assert_eq!(members(&after, parent), [(91, 3, b"keep".to_vec())]);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("keep").unwrap().read_string().unwrap(),
        ["keep"]
    );
    match edit {
        Edit::RemoveAttribute => {
            let dataset = file.dataset("value").unwrap();
            assert_eq!(dataset.read_i32().unwrap(), [17]);
            assert!(dataset.attrs().unwrap().is_empty());
        }
        Edit::ReplaceAttribute => {
            let dataset = file.dataset("value").unwrap();
            assert_eq!(dataset.read_i32().unwrap(), [17]);
            assert_eq!(
                dataset.attrs().unwrap().remove("a"),
                Some(AttrValue::I64(9))
            );
        }
        Edit::Overwrite => assert_eq!(
            file.dataset("value").unwrap().read_string().unwrap(),
            ["after"]
        ),
        Edit::Delete => assert_eq!(file.root().datasets().unwrap(), ["keep"]),
    }
}

#[rstest]
#[case::opaque(UnknownStorage::Opaque)]
#[case::unlinked_header(UnknownStorage::Header)]
#[case::unlinked_parent(UnknownStorage::Parent)]
fn public_commit_preserves_unknown_allocations_while_removing_an_attribute(
    #[case] unknown: UnknownStorage,
) {
    let path = test_util::temp::temp_path("public-unknown-heap-storage.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"removed"), Object::new(91, b"keep")]);
    let mut records = Form::CompactDataset.messages(
        &mut fixture,
        &hdf5_pure::make_i32_type(),
        &17_i32.to_le_bytes(),
    );
    records.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(7, parent, 7),
    ));
    fixture.object("value", records);
    let records = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(4, parent, 91),
    );
    fixture.object("keep", records);
    let unknown = unknown.append(&mut fixture, parent);
    fixture.write(&path);
    let original = fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.dataset("value").unwrap().remove_attr("a").unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let after = fs::read(&path).unwrap();
    assert_eq!(&after[unknown.clone()], &original[unknown]);
    let start = usize::try_from(parent).unwrap();
    let end = start.checked_add(MIN_COLLECTION_SIZE).unwrap();
    assert_eq!(&after[start..end], &original[start..end]);
    assert_eq!(
        members(&after, parent),
        [(7, 1, b"removed".to_vec()), (91, 1, b"keep".to_vec())]
    );
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("value").unwrap().read_i32().unwrap(), [17]);
    assert!(file.dataset("value").unwrap().attrs().unwrap().is_empty());
    assert_eq!(
        file.dataset("keep").unwrap().read_string().unwrap(),
        ["keep"]
    );
}

#[test]
fn public_commit_keeps_an_unobserved_member_after_scalar_fill_deletion() {
    let path = test_util::temp::temp_path("public-unobserved-fill-member.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[
        Object::new(7, b"removed"),
        Object {
            index: 91,
            reference_count: 0,
            data: b"unobserved",
        },
    ]);
    let records = Form::FillValue.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(7, parent, 7),
    );
    fixture.object("value", records);
    let records = Form::CompactDataset.messages(
        &mut fixture,
        &hdf5_pure::make_i32_type(),
        &17_i32.to_le_bytes(),
    );
    fixture.object("keep", records);
    fixture.write(&path);
    let original = fs::read(&path).unwrap();
    let retained = member_record(&original, parent, 91);
    let file = File::open_rw(&path).unwrap();
    assert_eq!(
        file.dataset("value").unwrap().read_string().unwrap(),
        ["removed"]
    );
    file.root().delete("value").unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let after = fs::read(&path).unwrap();
    assert_eq!(members(&after, parent), [(91, 0, b"unobserved".to_vec())]);
    assert_eq!(member_record(&after, parent, 91), retained);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), ["keep"]);
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [17]);
}

/// Copies the complete allocated record for `index` from the parsed parent.
///
/// # Panics
///
/// Panics if the parent or record is invalid or the index is absent.
fn member_record(bytes: &[u8], parent: u64, index: u16) -> Vec<u8> {
    let range = FileRange::new(
        PhysicalAddress::new(parent),
        u64::try_from(MIN_COLLECTION_SIZE).unwrap(),
    )
    .unwrap();
    let region = FileRegion::new(
        range,
        &bytes[usize::try_from(range.start().get()).unwrap()
            ..usize::try_from(range.end().get()).unwrap()],
    )
    .unwrap();
    let locator = GlobalHeapCollectionLocator::new(range.start(), LengthWidth::Eight);
    let mut walk = locator
        .parse(region.subregion(locator.parse_range().unwrap()).unwrap())
        .unwrap()
        .walk();
    while let Some(range) = walk.parse_range() {
        if let Some(object) = walk.parse(region.subregion(range).unwrap()).unwrap()
            && object.index().get() == index
        {
            return region.subregion(object.range()).unwrap().bytes().to_vec();
        }
    }
    panic!("missing member {index}");
}

/// Parses member indices, stored reference counts, and payload bytes in collection order.
///
/// # Panics
///
/// Panics if the parent or any member is invalid.
fn members(bytes: &[u8], parent: u64) -> Vec<(u16, u16, Vec<u8>)> {
    let address = PhysicalAddress::new(parent);
    let range = FileRange::new(address, u64::try_from(MIN_COLLECTION_SIZE).unwrap()).unwrap();
    let start = usize::try_from(parent).unwrap();
    let end = usize::try_from(range.end().get()).unwrap();
    let region = FileRegion::new(range, &bytes[start..end]).unwrap();
    let locator = GlobalHeapCollectionLocator::new(address, LengthWidth::Eight);
    let mut walk = locator
        .parse(region.subregion(locator.parse_range().unwrap()).unwrap())
        .unwrap()
        .walk();
    let mut objects = Vec::new();
    while let Some(range) = walk.parse_range() {
        if let Some(object) = walk.parse(region.subregion(range).unwrap()).unwrap() {
            objects.push((
                object.index().get(),
                object.reference_count(),
                object.payload().parse(region).unwrap().to_vec(),
            ));
        }
    }
    objects
}

#[derive(Clone, Copy, Debug)]
enum Edit {
    Delete,
    Overwrite,
    RemoveAttribute,
    ReplaceAttribute,
}

#[derive(Clone, Copy, Debug)]
enum UnknownStorage {
    Header,
    Opaque,
    Parent,
}

impl UnknownStorage {
    /// Appends selected unknown storage and returns its complete byte range.
    ///
    /// # Panics
    ///
    /// Panics if fixture construction fails.
    fn append(self, fixture: &mut Fixture, parent: u64) -> Range<usize> {
        let start = fixture.image.as_bytes().len();
        match self {
            Self::Header => {
                let records = Form::CompactDataset.messages(
                    fixture,
                    &global_heap_fixture::vlen_string(),
                    &global_heap_fixture::reference(7, parent, 7),
                );
                fixture.append_header(records);
            }
            Self::Opaque => {
                fixture.image.append(&[0x5a; 17]);
            }
            Self::Parent => {
                fixture.collection(&[Object::new(3, b"unlinked")]);
            }
        }
        start..fixture.image.as_bytes().len()
    }
}
