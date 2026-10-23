use std::ops::Range;
use std::path::Path;

use hdf5_pure::AttrValue;
use hdf5_pure::Datatype;
use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::MaxExtent;
use hdf5_pure::UserblockSize;
use rstest::rstest;
use test_util::data_layout;
use test_util::dataspace;
use test_util::global_heap;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::temp;
use test_util::widths::Widths;
use test_util_hdf5::file_space;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;
use test_util_hdf5::paged;

#[rstest]
#[case::contiguous_dataset(Written::Contiguous)]
#[case::filtered_chunked_dataset(Written::FilteredChunks)]
#[case::variable_length_attribute(Written::Attribute)]
fn the_collections_of_a_deleted_object_are_reclaimed_and_reused_for_raw_data(
    #[case] written: Written,
    #[values(0, USERBLOCK_SIZE)] userblock: u64,
) {
    let path = temp::temp_path("reclaim_collection.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    written.add_to(&mut builder, "x");
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    builder.write(&path).unwrap();
    let collections = collections(&path);
    let file = File::open_rw(&path).unwrap();

    file.root().delete("x").unwrap();
    file.commit().unwrap();
    for collection in &collections {
        assert_reclaimed(&file, collection);
    }
    let raw = assert_reused_for_raw_data(&file, &collections[0]);
    file.close().unwrap();

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [1, 2, 3]);
    assert_eq!(file.dataset("raw").unwrap().read_u8().unwrap(), raw);
}

#[rstest]
#[case::attribute_of_strings(Form::CompactAttribute, global_heap_fixture::vlen_string(), &[0])]
#[case::sequences(Form::ContiguousDataset, global_heap_fixture::vlen_sequence(), &[0])]
#[case::compound_with_a_string_member(
    Form::ContiguousDataset,
    global_heap_fixture::with_integer(global_heap_fixture::vlen_string()),
    &[4]
)]
#[case::array_of_sequences(
    Form::CompactDataset,
    global_heap_fixture::array_of(global_heap_fixture::vlen_sequence()),
    &[0, 16]
)]
#[case::fill_value_of_strings(Form::FillValue, global_heap_fixture::vlen_string(), &[0])]
fn the_collections_of_each_kind_of_reference_are_reclaimed_and_reused_for_raw_data(
    #[case] form: Form,
    #[case] datatype: Datatype,
    #[case] slots: &[usize],
) {
    let mut fixture = Fixture::new();
    let collections: Vec<u64> = slots
        .iter()
        .map(|_| fixture.collection(&[Object::new(1, b"alpha"), Object::new(2, b"beta")]))
        .collect();
    let mut element = vec![0; datatype.type_size() as usize];
    let elements: Vec<u8> = [1, 2]
        .into_iter()
        .flat_map(|index| {
            for (&slot, &collection) in slots.iter().zip(&collections) {
                element[slot..slot + global_heap_fixture::REFERENCE_SIZE]
                    .copy_from_slice(&global_heap_fixture::reference(5, collection, index));
            }
            element.clone()
        })
        .collect();
    let messages = form.messages(&mut fixture, &datatype, &elements);
    fixture.object("x", messages);
    let kept = kept_dataset(&mut fixture);
    let path = temp::temp_path("reclaim_each_kind.h5");
    fixture.write(&path);
    let file = File::open_rw(&path).unwrap();

    file.root().delete("x").unwrap();
    file.commit().unwrap();

    let collections: Vec<Range<u64>> = collections.into_iter().map(minimum_collection).collect();
    for collection in &collections {
        assert_reclaimed(&file, collection);
    }
    let raw = assert_reused_for_raw_data(&file, &collections[0]);
    file.close().unwrap();

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), kept);
    assert_eq!(file.dataset("raw").unwrap().read_u8().unwrap(), raw);
}

#[test]
fn deleting_a_group_reclaims_the_collections_of_its_subtree() {
    let path = temp::temp_path("reclaim_group.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let group = file.root().create_group("g").unwrap();
    group
        .set_attr(
            "labels",
            AttrValue::VarLenStringArray(vec!["alpha".into(), "beta".into()]),
        )
        .unwrap();
    file.root().create_group("g/sub").unwrap();
    for name in ["g/x", "g/sub/y"] {
        file.root()
            .create_dataset(name, |b| {
                b.with_vlen_strings(&["alpha", "beta"]);
            })
            .unwrap();
    }
    file.root()
        .create_dataset("after", |b| {
            b.with_i32_data(&[4, 5, 6]);
        })
        .unwrap();
    file.commit().unwrap();
    drop(group);
    let liveness = file.__global_heap_liveness().unwrap();
    assert!(liveness.analysis_complete);
    let collections: Vec<Range<u64>> = liveness
        .referenced_collections
        .into_iter()
        .map(minimum_collection)
        .collect();
    assert_eq!(collections.len(), 3, "{collections:?}");

    file.root().delete("g").unwrap();
    file.commit().unwrap();

    for collection in &collections {
        assert_reclaimed(&file, collection);
    }
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [1, 2, 3]);
    assert_eq!(
        file.dataset("after").unwrap().read_i32().unwrap(),
        [4, 5, 6]
    );
}

#[rstest]
#[case::in_an_earlier_commit(false)]
#[case::in_the_same_commit(true)]
fn a_collection_a_remaining_copy_refers_to_is_kept(#[case] same_commit: bool) {
    let path = temp::temp_path("reclaim_shared_kept.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("a").with_vlen_strings(&STRINGS);
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    builder.write(&path).unwrap();
    let collections = collections(&path);
    let file = File::open_rw(&path).unwrap();

    file.copy("a", "b").unwrap();
    if !same_commit {
        file.commit().unwrap();
    }
    file.root().delete("a").unwrap();
    file.commit().unwrap();

    for collection in &collections {
        assert_kept(&file, collection);
    }
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("b").unwrap().read_string().unwrap(), STRINGS);
}

#[test]
fn a_collection_two_deleted_datasets_share_is_reclaimed() {
    let path = temp::temp_path("reclaim_shared_deleted.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("a").with_vlen_strings(&STRINGS);
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    builder.write(&path).unwrap();
    let collections = collections(&path);
    let file = File::open_rw(&path).unwrap();
    file.copy("a", "b").unwrap();
    file.commit().unwrap();

    file.root().delete("a").unwrap();
    file.root().delete("b").unwrap();
    file.commit().unwrap();

    for collection in &collections {
        assert_reclaimed(&file, collection);
    }
    file.close().unwrap();
}

#[test]
fn deleting_one_of_two_hard_links_keeps_the_collection() {
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"alpha"), Object::new(2, b"beta")]);
    let elements = [
        global_heap_fixture::reference(5, collection, 1),
        global_heap_fixture::reference(4, collection, 2),
    ]
    .concat();
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &elements,
    );
    let dataset = fixture.append_header(messages);
    fixture.link("d", dataset);
    fixture.link("d2", dataset);
    kept_dataset(&mut fixture);
    let path = temp::temp_path("reclaim_hard_links.h5");
    fixture.write(&path);
    let file = File::open_rw(&path).unwrap();

    file.root().delete("d").unwrap();
    file.commit().unwrap();

    assert_kept(&file, &minimum_collection(collection));
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d2").unwrap().read_string().unwrap(),
        ["alpha", "beta"]
    );
}

#[rstest]
#[case::object_references(Unreadable::ObjectReferences)]
#[case::data_past_the_end_of_the_file(Unreadable::DataPastTheEndOfTheFile)]
fn a_remaining_object_the_analysis_cannot_read_keeps_every_collection(
    #[case] unreadable: Unreadable,
) {
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"alpha")]);
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, collection, 1),
    );
    fixture.object("x", messages);
    fixture.object("unreadable", unreadable.messages());
    kept_dataset(&mut fixture);
    let path = temp::temp_path("reclaim_incomplete_remaining.h5");
    fixture.write(&path);
    let file = File::open_rw(&path).unwrap();

    file.root().delete("x").unwrap();
    file.commit().unwrap();

    assert_kept(&file, &minimum_collection(collection));
    file.close().unwrap();
}

#[test]
fn a_deleted_object_the_analysis_cannot_read_keeps_every_collection() {
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"alpha")]);
    let mut messages = Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, collection, 1),
    );
    messages.extend(Unreadable::DataPastTheEndOfTheFile.messages());
    fixture.object("x", messages);
    kept_dataset(&mut fixture);
    let path = temp::temp_path("reclaim_incomplete_deleted.h5");
    fixture.write(&path);
    let file = File::open_rw(&path).unwrap();

    file.root().delete("x").unwrap();
    file.commit().unwrap();

    assert_kept(&file, &minimum_collection(collection));
    file.close().unwrap();
}

#[test]
fn an_unreadable_chunk_of_the_deleted_dataset_keeps_every_collection() {
    let path = temp::temp_path("reclaim_unreadable_chunk.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("x")
        .with_vlen_strings(&STRINGS)
        .with_chunks(&[2])
        .with_deflate(DeflateLevel::Six)
        .set_attr(
            "labels",
            AttrValue::VarLenStringArray(vec!["alpha".into(), "beta".into()]),
        );
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    builder.write(&path).unwrap();
    let collections = collections(&path);
    let chunk = File::open(&path)
        .unwrap()
        .dataset("x")
        .unwrap()
        .chunks()
        .unwrap()[0]
        .clone();
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[chunk.address as usize..(chunk.address + chunk.storage_size) as usize].fill(0xff);
    std::fs::write(&path, bytes).unwrap();
    let file = File::open_rw(&path).unwrap();

    file.root().delete("x").unwrap();
    file.commit().unwrap();

    for collection in &collections {
        assert_kept(&file, collection);
    }
    file.close().unwrap();
}

#[rstest]
#[case::version_2(global_heap::VERSION_OFFSET, &[2])]
#[case::unaligned_size(global_heap::SIZE_OFFSET, &(global_heap::MIN_COLLECTION_SIZE as u64 + 4).to_le_bytes())]
fn a_malformed_collection_is_kept_and_a_well_formed_one_reclaimed(
    #[case] offset: usize,
    #[case] bytes: &[u8],
) {
    let mut fixture = Fixture::new();
    let malformed = fixture.collection(&[Object::new(1, b"alpha")]);
    fixture.image.place(malformed as usize + offset, bytes);
    let well_formed = fixture.collection(&[Object::new(1, b"beta")]);
    let elements = [
        global_heap_fixture::reference(5, malformed, 1),
        global_heap_fixture::reference(4, well_formed, 1),
    ]
    .concat();
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &elements,
    );
    fixture.object("x", messages);
    kept_dataset(&mut fixture);
    let path = temp::temp_path("reclaim_malformed.h5");
    fixture.write(&path);
    let file = File::open_rw(&path).unwrap();

    file.root().delete("x").unwrap();
    file.commit().unwrap();

    assert_kept(&file, &minimum_collection(malformed));
    assert_reclaimed(&file, &minimum_collection(well_formed));
    file.close().unwrap();
}

#[test]
fn a_collection_reclaimed_in_a_paged_file_is_reused_in_place_by_raw_data() {
    let path = temp::temp_path("reclaim_paged.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("x").with_vlen_strings(&STRINGS);
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    builder
        .with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(FileSpacePageSize::try_from(PAGE_SIZE).unwrap());
    builder.write(&path).unwrap();
    let collections = collections(&path);
    let [collection] = &collections[..] else {
        panic!("expected one collection: {collections:?}");
    };
    let file = File::open_rw(&path).unwrap();

    file.root().delete("x").unwrap();
    file.commit().unwrap();
    assert_reclaimed(&file, collection);
    file.root()
        .create_dataset("raw", |b| {
            b.with_u8_data(&vec![7; (collection.end - collection.start) as usize]);
        })
        .unwrap();
    file.commit().unwrap();
    assert_eq!(file_space::contiguous_extent(&file, "raw"), *collection);
    file.close().unwrap();

    paged::assert_consistent(&path, PAGE_SIZE);
    paged::assert_pages_homogeneous(&path, PAGE_SIZE, &["keep", "raw"]);
}

#[derive(Clone, Copy, Debug)]
enum Written {
    Attribute,
    Contiguous,
    FilteredChunks,
}

impl Written {
    fn add_to(self, builder: &mut FileBuilder, name: &str) {
        let dataset = builder.create_dataset(name);
        match self {
            Self::Attribute => {
                dataset.with_i32_data(&[1, 2]).set_attr(
                    "labels",
                    AttrValue::VarLenStringArray(vec!["alpha".into(), "beta".into()]),
                );
            }
            Self::Contiguous => {
                dataset.with_vlen_strings(&STRINGS);
            }
            Self::FilteredChunks => {
                dataset
                    .with_vlen_strings(&STRINGS)
                    .with_chunks(&[2])
                    .with_deflate(DeflateLevel::Six)
                    .with_maxshape(&[MaxExtent::Unlimited]);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Unreadable {
    DataPastTheEndOfTheFile,
    ObjectReferences,
}

impl Unreadable {
    fn messages(self) -> Vec<Message> {
        let (datatype, layout) = match self {
            Self::DataPastTheEndOfTheFile => (
                global_heap_fixture::vlen_string(),
                data_layout::contiguous(
                    PAST_THE_END_OF_THE_FILE,
                    global_heap_fixture::REFERENCE_SIZE as u64,
                ),
            ),
            Self::ObjectReferences => (
                global_heap_fixture::object_reference(),
                data_layout::contiguous(global_heap_fixture::UNDEFINED_ADDRESS, 0),
            ),
        };
        let mut messages = global_heap_fixture::dataset_messages(&datatype, &dataspace::scalar());
        messages.push(Message::new(MessageType::DATA_LAYOUT, &layout));
        messages
    }
}

fn kept_dataset(fixture: &mut Fixture) -> Vec<u8> {
    let values = vec![1, 2, 3, 4];
    let mut messages = global_heap_fixture::dataset_messages(
        &hdf5_pure::make_u8_type(),
        &dataspace::v2(
            1,
            dataspace::Flags::NONE,
            dataspace::Kind::SIMPLE,
            &[values.len() as u64],
            None,
        ),
    );
    let address = fixture.image.append(&values) as u64;
    messages.push(Message::new(
        MessageType::DATA_LAYOUT,
        &data_layout::contiguous(address, values.len() as u64),
    ));
    fixture.object("keep", messages);
    values
}

fn collections(path: &Path) -> Vec<Range<u64>> {
    global_heap::collection_ranges(&std::fs::read(path).unwrap(), Widths::EIGHT)
}

fn minimum_collection(start: u64) -> Range<u64> {
    start..start + global_heap::MIN_COLLECTION_SIZE as u64
}

#[track_caller]
fn assert_reused_for_raw_data(file: &File, collection: &Range<u64>) -> Vec<u8> {
    let raw = vec![7; (collection.end - collection.start) as usize];
    file.root()
        .create_dataset("raw", |b| {
            b.with_u8_data(&raw);
        })
        .unwrap();
    file.commit().unwrap();
    let placed = file_space::contiguous_extent(file, "raw");
    assert!(
        placed.start < collection.end && collection.start < placed.end,
        "the raw data at {placed:?} does not reuse the collection at {collection:?}"
    );
    raw
}

#[track_caller]
fn assert_reclaimed(file: &File, collection: &Range<u64>) {
    let accounting = file.space_accounting().unwrap();
    let reusable = accounting
        .reusable_free_space
        .iter()
        .any(|&(address, len)| address <= collection.start && collection.end <= address + len);
    assert!(
        reusable || file.file_size() <= collection.start,
        "the collection at {collection:?} is neither reusable in {:?} nor past the end of the \
         file at {}",
        accounting.reusable_free_space,
        file.file_size()
    );
}

#[track_caller]
fn assert_kept(file: &File, collection: &Range<u64>) {
    let accounting = file.space_accounting().unwrap();
    assert!(
        accounting
            .reusable_free_space
            .iter()
            .all(|&(address, len)| address + len <= collection.start || collection.end <= address),
        "the collection at {collection:?} overlaps the reusable space {:?}",
        accounting.reusable_free_space
    );
    assert!(collection.end <= file.file_size());
}

const STRINGS: [&str; 3] = ["alpha", "", "beta"];
const USERBLOCK_SIZE: u64 = 512;
const PAGE_SIZE: u64 = 4096;
const PAST_THE_END_OF_THE_FILE: u64 = 1 << 40;
