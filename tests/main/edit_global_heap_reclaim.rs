use std::ops::Range;
use std::path::Path;

use hdf5_pure::AttrValue;
use hdf5_pure::Datatype;
use hdf5_pure::DeflateLevel;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileLocking;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FormatError;
use hdf5_pure::FreeSections;
use hdf5_pure::MaxExtent;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::PhysicalAddress;
use hdf5_pure::UserblockSize;
use rstest::rstest;
use test_util::data_layout;
use test_util::dataspace;
use test_util::global_heap;
use test_util::global_heap::Object;
use test_util::group_info;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v2;
use test_util::temp;
use test_util::widths::Widths;
use test_util_hdf5::file_space;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;
use test_util_hdf5::paged;

#[rstest]
fn damaged_collection_does_not_release_a_residual_groups_predecessor(
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let mut fixture = Fixture::new();
    let damaged = fixture.collection(&[Object::new(7, b"broken")]);
    fixture.image.place(
        usize::try_from(damaged).unwrap() + global_heap::VERSION_OFFSET,
        &[2],
    );
    let victim = fixture.collection(&[Object::new(7, b"removed")]);
    for (name, parent, length) in [("damaged", damaged, 6), ("victim", victim, 7)] {
        let messages = Form::CompactDataset.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(length, parent, 7),
        );
        fixture.object(name, messages);
    }
    let messages = Form::CompactDataset.messages(
        &mut fixture,
        &hdf5_pure::make_i32_type(),
        &17_i32.to_le_bytes(),
    );
    let keeper = fixture.append_header(messages);
    fixture.link("keeper", keeper);
    let header_end = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let group = v2::Header::new()
        .messages(group_info::compact_messages(&[
            global_heap_fixture::hard_link("child", keeper),
        ]))
        .build();
    let residual = u64::try_from(fixture.image.append(&group)).unwrap();
    let residual_end = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    assert_eq!(header_end, residual);
    let damaged_end = damaged + u64::try_from(global_heap::MIN_COLLECTION_SIZE).unwrap();
    let protected = [
        damaged..damaged_end,
        keeper..header_end,
        residual..residual_end,
    ];
    let path = temp::temp_path("damaged-collection-residual-keeper.h5");
    fixture.write(&path);

    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_locking(FileLocking::Disabled)
            .with_memory_strategy(memory),
    )
    .unwrap();
    let before = std::fs::read(&path).unwrap();
    assert_eq!(
        &before[usize::try_from(residual).unwrap()..usize::try_from(residual_end).unwrap()],
        group
    );
    assert!(matches!(
        file.dataset("damaged").unwrap().read_string(),
        Err(Error::Format(FormatError::InvalidGlobalHeapVersion(2)))
    ));
    assert_eq!(
        file.dataset("victim").unwrap().read_string().unwrap(),
        ["removed"]
    );
    assert_eq!(file.dataset("keeper").unwrap().read_i32().unwrap(), [17]);

    file.root().delete("victim").unwrap();
    file.dataset("keeper")
        .unwrap()
        .set_attr("a", AttrValue::I64(9))
        .unwrap();
    file.commit().unwrap();
    assert!(!file.has_staged_edits());
    let after = std::fs::read(&path).unwrap();
    for range in &protected {
        assert_kept(&file, range);
        let bytes = usize::try_from(range.start).unwrap()..usize::try_from(range.end).unwrap();
        assert_eq!(&after[bytes.clone()], &before[bytes]);
    }

    let raw = [31; 256];
    file.root()
        .create_dataset("added", |dataset| {
            dataset.with_u8_data(&raw);
        })
        .unwrap();
    file.commit().unwrap();
    assert!(!file.has_staged_edits());
    let placed = file_space::contiguous_extent(&file, "added");
    for range in &protected {
        assert_kept(&file, range);
        assert!(placed.end <= range.start || range.end <= placed.start);
    }
    file.close().unwrap();
    let after = std::fs::read(&path).unwrap();
    for range in protected {
        let bytes = usize::try_from(range.start).unwrap()..usize::try_from(range.end).unwrap();
        assert_eq!(&after[bytes.clone()], &before[bytes]);
    }
    let file = File::open(&path).unwrap();
    let mut datasets = file.root().datasets().unwrap();
    datasets.sort();
    assert_eq!(datasets, ["added", "damaged", "keeper"]);
    assert_eq!(file.dataset("added").unwrap().read_u8().unwrap(), raw);
    assert!(matches!(
        file.dataset("damaged").unwrap().read_string(),
        Err(Error::Format(FormatError::InvalidGlobalHeapVersion(2)))
    ));
    let keeper = file.dataset("keeper").unwrap();
    assert_eq!(keeper.read_i32().unwrap(), [17]);
    assert_eq!(keeper.attrs().unwrap()["a"], AttrValue::I64(9));
}

#[rstest]
fn partial_append_retires_the_original_tail_with_complete_release_evidence(
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let source = partial_append_source();
    let path = temp::temp_path("captured-partial-append.h5");
    source.fixture.write(&path);
    let before = std::fs::read(&path).unwrap();
    let file = append_to_partial_chunk(&path, memory, &before, &source.tail);

    let protected = [minimum_collection(source.neighbor)];
    assert_protected_bytes(&file, &path, &before, &protected);
    assert_reclaimed(&file, &source.tail);

    let file = append_followup_dataset(file, &path, &before, &protected);
    assert_eq!(
        file.dataset("neighbor").unwrap().read_string().unwrap(),
        ["nearby"]
    );
}

#[rstest]
fn partial_append_preserves_the_original_tail_when_release_evidence_is_unavailable(
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let mut source = partial_append_source();
    source.fixture.image.place(
        usize::try_from(source.neighbor).unwrap() + global_heap::VERSION_OFFSET,
        &[2],
    );
    let keeper = source.keeper;
    let group = v2::Header::new()
        .messages(group_info::compact_messages(&[
            global_heap_fixture::hard_link("child", keeper.start),
        ]))
        .build();
    let residual = u64::try_from(source.fixture.image.append(&group)).unwrap();
    let residual_end = u64::try_from(source.fixture.image.as_bytes().len()).unwrap();
    let path = temp::temp_path("captured-partial-append.h5");
    source.fixture.write(&path);
    let before = std::fs::read(&path).unwrap();
    let file = append_to_partial_chunk(&path, memory, &before, &source.tail);

    let protected = [
        minimum_collection(source.neighbor),
        keeper,
        residual..residual_end,
        source.tail,
    ];
    assert_protected_bytes(&file, &path, &before, &protected);

    let file = append_followup_dataset(file, &path, &before, &protected);
    assert!(matches!(
        file.dataset("neighbor").unwrap().read_string(),
        Err(Error::Format(FormatError::InvalidGlobalHeapVersion(2)))
    ));
}

struct PartialAppendSource {
    fixture: Fixture,
    neighbor: u64,
    keeper: Range<u64>,
    tail: Range<u64>,
}

fn partial_append_source() -> PartialAppendSource {
    let mut fixture = Fixture::new();
    let neighbor = fixture.collection(&[Object::new(7, b"nearby")]);
    let retired_heap = fixture.collection(&[Object::new(7, b"removed")]);
    for (name, parent, length) in [("neighbor", neighbor, 6), ("victim", retired_heap, 7)] {
        let messages = Form::CompactDataset.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(length, parent, 7),
        );
        fixture.object(name, messages);
    }

    let original_chunk = [17_i32, 18, 0, 0].map(i32::to_le_bytes).concat();
    let tail = u64::try_from(fixture.image.append(&original_chunk)).unwrap();
    let tail_end = tail + u64::try_from(original_chunk.len()).unwrap();
    let messages = Form::CompactDataset.messages(
        &mut fixture,
        &hdf5_pure::make_i32_type(),
        &23_i32.to_le_bytes(),
    );
    fixture.object("anchor", messages);
    let mut messages = global_heap_fixture::dataset_messages(
        &hdf5_pure::make_i32_type(),
        &dataspace::v2(
            1,
            dataspace::Flags::MAX_DIMS,
            dataspace::Kind::SIMPLE,
            &[2],
            Some(&[4]),
        ),
    );
    messages.push(Message::new(
        MessageType::DATA_LAYOUT,
        &data_layout::single_chunk(Widths::EIGHT, None, Some(tail)),
    ));
    let keeper = fixture.append_header(messages);
    fixture.link("keeper", keeper);
    let keeper_end = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    PartialAppendSource {
        fixture,
        neighbor,
        keeper: keeper..keeper_end,
        tail: tail..tail_end,
    }
}

fn append_to_partial_chunk(
    path: &Path,
    memory: MemoryStrategy,
    before: &[u8],
    original_tail: &Range<u64>,
) -> File {
    let tail = original_tail.start;
    let tail_end = original_tail.end;
    let file = File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_locking(FileLocking::Disabled)
            .with_memory_strategy(memory),
    )
    .unwrap();
    let original_chunk = [17_i32, 18, 0, 0].map(i32::to_le_bytes).concat();
    assert_eq!(&before[tail as usize..tail_end as usize], original_chunk);
    let [original]: [Range<u64>; 1] = file_space::raw_extents(&file, "keeper").try_into().unwrap();
    assert_eq!(original, tail..tail_end);
    assert_eq!(
        file.dataset("keeper").unwrap().read_i32().unwrap(),
        [17, 18]
    );
    assert_eq!(
        file.dataset("victim").unwrap().read_string().unwrap(),
        ["removed"]
    );

    file.root().delete("victim").unwrap();
    file.dataset("keeper")
        .unwrap()
        .append_staged(|append| {
            append.append_i32(&[19]);
        })
        .unwrap();
    file.commit().unwrap();
    assert!(!file.has_staged_edits());
    let [placed]: [Range<u64>; 1] = file_space::raw_extents(&file, "keeper").try_into().unwrap();
    assert!(placed.end <= tail || tail_end <= placed.start);
    let after = std::fs::read(path).unwrap();
    let replacement = [17_i32, 18, 19, 0].map(i32::to_le_bytes).concat();
    assert_eq!(
        &after[placed.start as usize..placed.end as usize],
        replacement
    );
    file
}

fn assert_protected_bytes(file: &File, path: &Path, before: &[u8], protected: &[Range<u64>]) {
    let after = std::fs::read(path).unwrap();
    for range in protected {
        assert_kept(file, range);
        let bytes = range.start as usize..range.end as usize;
        assert_eq!(&after[bytes.clone()], &before[bytes]);
    }
}

fn append_followup_dataset(
    file: File,
    path: &Path,
    before: &[u8],
    protected: &[Range<u64>],
) -> File {
    let raw = [31_u8; 16];
    file.root()
        .create_dataset("added", |dataset| {
            dataset.with_u8_data(&raw);
        })
        .unwrap();
    file.commit().unwrap();
    let added = file_space::contiguous_extent(&file, "added");
    for range in protected {
        assert_kept(&file, range);
        assert!(added.end <= range.start || range.end <= added.start);
    }
    file.close().unwrap();
    let after = std::fs::read(path).unwrap();
    for range in protected {
        let bytes = range.start as usize..range.end as usize;
        assert_eq!(&after[bytes.clone()], &before[bytes]);
    }

    let file = File::open(path).unwrap();
    assert_eq!(
        file.dataset("keeper").unwrap().read_i32().unwrap(),
        [17, 18, 19]
    );
    assert_eq!(file.dataset("anchor").unwrap().read_i32().unwrap(), [23]);
    assert_eq!(file.dataset("added").unwrap().read_u8().unwrap(), raw);
    assert!(
        !file
            .root()
            .datasets()
            .unwrap()
            .iter()
            .any(|name| name == "victim")
    );
    file
}

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
    let count = if matches!(form, Form::FillValue) {
        1
    } else {
        2
    };
    let objects = [Object::new(1, b"alpha"), Object::new(2, b"beta")];
    let collections: Vec<u64> = slots
        .iter()
        .map(|_| fixture.collection(&objects[..count]))
        .collect();
    let mut element = vec![0; datatype.type_size() as usize];
    let elements: Vec<u8> = (1..=u32::try_from(count).unwrap())
        .flat_map(|index| {
            for (&slot, &collection) in slots.iter().zip(&collections) {
                element[slot..slot + global_heap_fixture::REFERENCE_SIZE]
                    .copy_from_slice(&global_heap_fixture::reference(5, collection, index));
            }
            element.clone()
        })
        .collect();
    let messages = form.messages(&mut fixture, &datatype, &elements);
    fixture.object(
        "x",
        if matches!(form, Form::CompactAttribute) {
            group_info::compact_messages(&messages)
        } else {
            messages
        },
    );
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
    file.close().unwrap();
    let collections = collections(&path);
    let file = File::open_rw(&path).unwrap();
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

#[rstest]
#[case::replaced(Change::Replace)]
#[case::removed(Change::Remove)]
#[case::owner_deleted(Change::DeleteOwner)]
fn a_collection_string_attributes_share_is_kept_while_one_refers_to_it(#[case] change: Change) {
    let path = temp::temp_path("reclaim_shared_attribute.h5");
    let mut builder = FileBuilder::new();
    builder.set_attr("kept", strings(&["alpha", "beta"]));
    builder
        .create_dataset("x")
        .with_i32_data(&[1, 2])
        .set_attr("changed", strings(&["gamma", "delta"]));
    builder.write(&path).unwrap();
    let collections = collections(&path);
    assert_eq!(collections.len(), 1, "the attributes share one collection");
    let file = File::open_rw(&path).unwrap();

    change.apply(&file);
    file.commit().unwrap();

    assert_kept(&file, &collections[0]);
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.root().attrs().unwrap()["kept"],
        strings(&["alpha", "beta"])
    );
    if matches!(change, Change::Replace) {
        assert_eq!(
            file.dataset("x").unwrap().attrs().unwrap()["changed"],
            strings(&["epsilon"])
        );
    }
}

#[test]
fn a_collection_the_string_attributes_of_two_deleted_datasets_share_is_reclaimed() {
    let path = temp::temp_path("reclaim_shared_attribute_deleted.h5");
    let mut builder = FileBuilder::new();
    for (name, value) in [("x", "alpha"), ("y", "beta")] {
        builder
            .create_dataset(name)
            .with_i32_data(&[1, 2])
            .set_attr("label", strings(&[value]));
    }
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    builder.write(&path).unwrap();
    let collections = collections(&path);
    assert_eq!(collections.len(), 1, "the attributes share one collection");
    let file = File::open_rw(&path).unwrap();

    file.root().delete("x").unwrap();
    file.root().delete("y").unwrap();
    file.commit().unwrap();

    assert_reclaimed(&file, &collections[0]);
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
    let mut messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &elements,
    );
    messages.push(Message::reference_count(2));
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

#[test]
fn a_detached_group_keeps_a_deleted_groups_descendant_allocations() {
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"alpha")]);
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, collection, 1),
    );
    let dataset = fixture.append_header(messages);
    let child = fixture.append_header(group_info::compact_messages(&[
        global_heap_fixture::hard_link("strings", dataset),
    ]));
    let group = fixture.append_header(group_info::compact_messages(&[
        global_heap_fixture::hard_link("child", child),
    ]));
    fixture.link("g", group);
    fixture.append_header(group_info::compact_messages(&[
        global_heap_fixture::hard_link("original", group),
    ]));
    let end = fixture.image.as_bytes().len();
    let kept = kept_dataset(&mut fixture);
    let path = temp::temp_path("reclaim_detached_group.h5");
    fixture.write(&path);
    let before = std::fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    assert_eq!(
        file.dataset("g/child/strings")
            .unwrap()
            .read_string()
            .unwrap(),
        ["alpha"]
    );

    file.root().delete("g").unwrap();
    file.commit().unwrap();

    // These allocations are contiguous in the fixture, from the heap through the detached header.
    assert_kept(&file, &(collection..u64::try_from(end).unwrap()));
    file.close().unwrap();
    let range = usize::try_from(collection).unwrap()..end;
    let after = std::fs::read(&path).unwrap();
    assert_eq!(&after[range.clone()], &before[range]);
    let file = File::open(&path).unwrap();
    assert!(file.root().groups().unwrap().is_empty());
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), kept);
    file.close().unwrap();
}

#[test]
fn remaining_object_references_preserve_the_source_collection() {
    let (fixture, collection) = remaining_references_source(Unreadable::ObjectReferences);
    let path = temp::temp_path("reclaim_incomplete_remaining.h5");
    fixture.write(&path);
    let file = File::open_rw(&path).unwrap();

    file.root().delete("x").unwrap();
    file.commit().unwrap();

    assert_kept(&file, &minimum_collection(collection));
    file.close().unwrap();
}

#[test]
fn remaining_data_past_source_eof_rejects_commit_without_changing_bytes() {
    let (fixture, collection) = remaining_references_source(Unreadable::DataPastTheEndOfTheFile);
    let path = temp::temp_path("reclaim_incomplete_remaining.h5");
    fixture.write(&path);
    let original = std::fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();

    file.root().delete("x").unwrap();
    assert_source_eof(
        file.commit().unwrap_err(),
        u64::try_from(original.len()).unwrap(),
    );

    assert_kept(&file, &minimum_collection(collection));
    file.close().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

fn remaining_references_source(unreadable: Unreadable) -> (Fixture, u64) {
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
    (fixture, collection)
}

#[test]
fn deleted_object_source_bounds_reject_commit_without_changing_bytes() {
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
    let original = std::fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();

    file.root().delete("x").unwrap();
    assert_source_eof(
        file.commit().unwrap_err(),
        u64::try_from(original.len()).unwrap(),
    );

    assert_kept(&file, &minimum_collection(collection));
    file.close().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), original);
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
fn unsupported_parent_geometry_keeps_both_collections(#[case] offset: usize, #[case] bytes: &[u8]) {
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
    let before = std::fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let Error::Format(error) = file.dataset("x").unwrap().read_string().unwrap_err() else {
        panic!("expected a malformed heap error");
    };
    assert_eq!(
        error,
        if offset == global_heap::VERSION_OFFSET {
            FormatError::InvalidGlobalHeapVersion(2)
        } else {
            FormatError::UnalignedGlobalHeapCollection {
                collection_address: malformed,
                size: global_heap::MIN_COLLECTION_SIZE as u64 + 4,
            }
        },
    );

    file.root().delete("x").unwrap();
    file.commit().unwrap();

    let ranges = [
        minimum_collection(malformed),
        minimum_collection(well_formed),
    ];
    for range in &ranges {
        assert_kept(&file, range);
    }
    file.close().unwrap();
    let after = std::fs::read(&path).unwrap();
    for range in ranges {
        let range = usize::try_from(range.start).unwrap()..usize::try_from(range.end).unwrap();
        assert_eq!(&after[range.clone()], &before[range]);
    }
}

#[test]
fn a_collection_reclaimed_in_a_paged_file_is_reused_in_place_by_raw_data() {
    let path = temp::temp_path("reclaim_paged.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("x").with_vlen_strings(&STRINGS);
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(0)
            .with_page_size(FileSpacePageSize::try_from(PAGE_SIZE).unwrap()),
    );
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

/// A change to the attribute `changed` of dataset `x`, whose strings share a collection with the
/// root attribute `kept`.
#[derive(Clone, Copy, Debug)]
enum Change {
    /// Deletes the dataset.
    DeleteOwner,
    /// Removes the attribute.
    Remove,
    /// Replaces the strings of the attribute, which go to a collection of their own.
    Replace,
}

impl Change {
    fn apply(self, file: &File) {
        match self {
            Self::DeleteOwner => file.root().delete("x").unwrap(),
            Self::Remove => file.dataset("x").unwrap().remove_attr("changed").unwrap(),
            Self::Replace => file
                .dataset("x")
                .unwrap()
                .set_attr("changed", strings(&["epsilon"]))
                .unwrap(),
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

fn assert_source_eof(error: Error, available: u64) {
    let Error::Format(FormatError::UnexpectedSourceEof {
        expected,
        available: actual,
    }) = error
    else {
        panic!("unexpected source-bound error: {error:?}");
    };
    assert_eq!(
        expected,
        PAST_THE_END_OF_THE_FILE
            .checked_add(u64::try_from(global_heap_fixture::REFERENCE_SIZE).unwrap())
            .unwrap()
    );
    assert_eq!(actual, available);
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

fn strings(values: &[&str]) -> AttrValue {
    AttrValue::VarLenStringArray(values.iter().map(|&value| value.to_owned()).collect())
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
    let sections = file.reusable_free_sections().unwrap();
    let _: &FreeSections<PhysicalAddress> = &sections;
    let reusable = sections.sections().iter().any(|section| {
        section.address().get() <= collection.start && collection.end <= section.end_address().get()
    });
    assert!(
        reusable || file.file_size() <= collection.start,
        "the collection at {collection:?} is neither reusable in {:?} nor past the end of the \
         file at {}",
        sections,
        file.file_size()
    );
}

#[track_caller]
fn assert_kept(file: &File, collection: &Range<u64>) {
    let sections = file.reusable_free_sections().unwrap();
    let _: &FreeSections<PhysicalAddress> = &sections;
    assert!(
        sections
            .sections()
            .iter()
            .all(|section| section.end_address().get() <= collection.start
                || collection.end <= section.address().get()),
        "the collection at {collection:?} overlaps the reusable space {:?}",
        sections
    );
    assert!(collection.end <= file.file_size());
}

const STRINGS: [&str; 3] = ["alpha", "", "beta"];
const USERBLOCK_SIZE: u64 = 512;
const PAGE_SIZE: u64 = 4096;
const PAST_THE_END_OF_THE_FILE: u64 = 1 << 40;
