use std::mem;
use std::path::Path;
use std::sync::{Arc, Mutex};

use hdf5_pure_core::__private::FileRange;
use rstest::rstest;
use test_util::global_heap::{MIN_COLLECTION_SIZE, Object};
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::{File, FileAccessProperties, FileInner};
use crate::edit::open_observed_engine;
use crate::{AttrValue, Layout, MetadataCacheConfig, PhysicalAddress, SyncPolicy};

#[rstest]
#[case::group(Edit::Group)]
#[case::numeric_attribute(Edit::Attribute)]
#[case::numeric_overwrite(Edit::Overwrite)]
fn candidate_free_public_commits_skip_unrelated_payload_reads(#[case] edit: Edit) {
    let path = test_util::temp::temp_path("candidate-free-commit-reads.h5");
    let parent = fixture(&path);
    let reads = Arc::new(Mutex::new(Vec::new()));
    let engine = open_observed_engine(&path, Arc::clone(&reads));
    let file = File {
        inner: Arc::new(
            FileInner::from_rw_session(
                engine,
                FileAccessProperties::new()
                    .with_metadata_cache(MetadataCacheConfig::disabled())
                    .with_sync_policy(SyncPolicy::OnClose),
            )
            .unwrap(),
        ),
    };
    let large = extent(&file, "large");
    assert_eq!(large.length(), u64::try_from(LARGE).unwrap());
    let labels = extent(&file, "labels");
    assert_eq!(labels.length(), 16);
    let old_value = extent(&file, "value");
    assert_eq!(old_value.length(), 4);
    let unrelated = [parent, labels, large];

    reads.lock().unwrap().clear();
    assert_eq!(
        file.dataset("labels").unwrap().read_string().unwrap(),
        ["retained"]
    );
    assert_eq!(
        file.dataset("large")
            .unwrap()
            .read_u8_region(&[0], &[1])
            .unwrap(),
        [0x2a]
    );
    let control = mem::take(&mut *reads.lock().unwrap());
    for range in unrelated {
        assert!(control.iter().any(|read| read.overlaps(range)));
    }

    match edit {
        Edit::Group => {
            file.root().create_group("added").unwrap();
        }
        Edit::Attribute => file
            .dataset("value")
            .unwrap()
            .set_attr("a", AttrValue::I64(9))
            .unwrap(),
        Edit::Overwrite => file
            .dataset("value")
            .unwrap()
            .write_staged(|builder| {
                builder.with_i32_data(&[11]);
            })
            .unwrap(),
    }
    reads.lock().unwrap().clear();
    file.commit().unwrap();
    let requested = mem::take(&mut *reads.lock().unwrap());
    let bytes = requested.iter().map(|read| read.length()).sum::<u64>();
    assert!(
        (1..=64 * 1024).contains(&bytes),
        "commit requested {bytes} bytes"
    );
    for range in unrelated {
        assert!(
            requested.iter().all(|read| !read.overlaps(range)),
            "commit read {range:?}: {requested:?}"
        );
    }
    if matches!(edit, Edit::Overwrite) {
        assert!(requested.iter().any(|read| read.contains(old_value)));
        assert_eq!(extent(&file, "value"), old_value);
    }
    assert!(!file.has_staged_edits());
    file.close().unwrap();

    let file = File::open(&path).unwrap();
    let dataset = file.dataset("value").unwrap();
    let expected = if matches!(edit, Edit::Overwrite) {
        11
    } else {
        7
    };
    assert_eq!(dataset.read_i32().unwrap(), [expected]);
    if matches!(edit, Edit::Attribute) {
        assert_eq!(dataset.attrs().unwrap()["a"], AttrValue::I64(9));
    } else {
        assert!(dataset.attrs().unwrap().is_empty());
    }
    let groups = file.root().groups().unwrap();
    if matches!(edit, Edit::Group) {
        assert_eq!(groups, ["added"]);
    } else {
        assert!(groups.is_empty());
    }
    assert_eq!(
        file.dataset("labels").unwrap().read_string().unwrap(),
        ["retained"]
    );
    assert_eq!(
        file.dataset("large").unwrap().read_u8().unwrap(),
        vec![0x2a; LARGE]
    );
}

#[derive(Clone, Copy, Debug)]
enum Edit {
    Group,
    Attribute,
    Overwrite,
}

fn fixture(path: &Path) -> FileRange {
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"retained")]);
    let labels = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(8, parent, 7),
    );
    fixture.object("labels", labels);
    let large =
        Form::ContiguousDataset.messages(&mut fixture, &crate::make_u8_type(), &vec![0x2a; LARGE]);
    fixture.object("large", large);
    let value = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::make_i32_type(),
        &7_i32.to_le_bytes(),
    );
    fixture.object("value", value);
    fixture.write(path);
    FileRange::new(
        PhysicalAddress::new(parent),
        u64::try_from(MIN_COLLECTION_SIZE).unwrap(),
    )
    .unwrap()
}

fn extent(file: &File, name: &str) -> FileRange {
    let Layout::Contiguous {
        address: Some(address),
        size,
    } = file.dataset(name).unwrap().layout().unwrap()
    else {
        panic!("{name} lacks allocated contiguous storage");
    };
    FileRange::new(PhysicalAddress::new(address), size).unwrap()
}

const LARGE: usize = 1 << 20;
