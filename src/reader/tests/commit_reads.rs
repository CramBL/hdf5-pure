use std::mem;
use std::path::Path;
use std::sync::{Arc, Mutex};

use hdf5_pure_core::__private::FileRange;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::global_heap::{self, MIN_COLLECTION_SIZE, Object};
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::{File, FileAccessProperties};
use crate::edit::open_observed_engine;
use crate::{
    AttrValue, FileBuilder, FileSpaceSettings, FileSpaceStrategy, Layout, MetadataCacheConfig,
    PhysicalAddress, SyncPolicy,
};

#[rstest]
#[case::group(Edit::Group)]
#[case::numeric_attribute(Edit::Attribute)]
#[case::numeric_overwrite(Edit::Overwrite)]
fn candidate_free_public_commits_skip_unrelated_payload_reads(#[case] edit: Edit) {
    let path = test_util::temp::temp_path("candidate-free-commit-reads.h5");
    let parent = fixture(&path);
    let reads = Arc::new(Mutex::new(Vec::new()));
    let file = open_observed_file(&path, &reads);
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

#[rstest]
#[case::reference_free(false)]
#[case::stored_root_reference(true)]
fn repeated_public_commits_read_untouched_headers_only_when_needed(#[case] references: bool) {
    let path = test_util::temp::temp_path("reference-walk-commit-reads.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::None));
    let probe = builder.create_dataset("probe");
    if references {
        probe.with_path_references(&[""]);
    } else {
        probe.with_u64_data(&[37]);
    }
    builder.create_dataset("original").with_i32_data(&[23, 29]);
    builder.write(&path).unwrap();

    let reads = Arc::new(Mutex::new(Vec::new()));
    let file = open_observed_file(&path, &reads);
    let addresses = ["probe", "original"]
        .map(|name| PhysicalAddress::new(file.dataset(name).unwrap().header_address().unwrap()));
    // Reading either untouched header starts at its signature.
    let headers = addresses.map(|address| FileRange::new(address, 4).unwrap());
    let payload = extent(&file, "probe");
    assert_eq!(payload.length(), 8);
    let mut root = file.root().header_address().unwrap();
    for index in 0..3 {
        file.root()
            .create_dataset(&format!("added{index}"), |builder| {
                builder.with_i32_data(&[index]);
            })
            .unwrap();
        reads.lock().unwrap().clear();
        file.commit().unwrap();
        let requested = mem::take(&mut *reads.lock().unwrap());
        for header in headers {
            assert_eq!(
                requested.iter().any(|read| read.overlaps(header)),
                references || index == 0,
                "commit {index} and header {header:?}: {requested:?}"
            );
        }
        assert_eq!(
            requested.iter().any(|read| read.overlaps(payload)),
            references,
            "commit {index} and payload {payload:?}: {requested:?}"
        );
        assert!(!file.has_staged_edits());
        let published = file.root().header_address().unwrap();
        assert_ne!(published, root);
        root = published;
    }
    file.close().unwrap();

    let file = File::open(&path).unwrap();
    for (name, address) in ["probe", "original"].into_iter().zip(addresses) {
        assert_eq!(
            file.dataset(name).unwrap().header_address().unwrap(),
            address.get()
        );
    }
    assert_eq!(file.root().header_address().unwrap(), root);
    assert_eq!(
        file.dataset("original").unwrap().read_i32().unwrap(),
        [23, 29]
    );
    for index in 0..3 {
        let dataset = file.dataset(&format!("added{index}")).unwrap();
        assert_eq!(dataset.read_i32().unwrap(), [index]);
    }
    let probe = file.dataset("probe").unwrap();
    if references {
        let resolved = probe.dereference().unwrap();
        let [crate::Object::Group(target)] = resolved.as_slice() else {
            panic!("expected a reference to the current root: {resolved:?}");
        };
        assert_eq!(target.header_address().unwrap(), root);
    } else {
        assert_eq!(probe.read_u64().unwrap(), [37]);
    }
}

#[test]
fn heap_bearing_commit_bounds_descriptor_reads_and_skips_numeric_payload() {
    let path = test_util::temp::temp_path("heap-bearing-commit-reads.h5");
    let mut fixture = Fixture::new();
    let retained = Object::new(91, b"retained");
    let parent = fixture.collection(&[Object::new(7, b"gone"), retained]);
    let datatype = global_heap_fixture::vlen_string();
    let labels = global_heap_fixture::reference(8, parent, 91).repeat(8_192);
    let messages = Form::ContiguousDataset.messages(&mut fixture, &datatype, &labels);
    fixture.object("labels", messages);
    let mut messages =
        Form::ContiguousDataset.messages(&mut fixture, &crate::make_u8_type(), &vec![0x2a; LARGE]);
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &datatype,
        &global_heap_fixture::reference(4, parent, 7),
    ));
    fixture.object("large", messages);
    fixture.write(&path);

    let reads = Arc::new(Mutex::new(Vec::new()));
    let file = open_observed_file(&path, &reads);
    let descriptors = extent(&file, "labels");
    let numeric = extent(&file, "large");
    assert_eq!(descriptors.length(), u64::try_from(labels.len()).unwrap());
    assert_eq!(numeric.length(), u64::try_from(LARGE).unwrap());
    let mut large = file.dataset("large").unwrap();
    reads.lock().unwrap().clear();
    assert_eq!(large.read_u8_region(&[0], &[1]).unwrap(), [0x2a]);
    assert!(
        reads
            .lock()
            .unwrap()
            .iter()
            .any(|read| read.overlaps(numeric))
    );

    large.remove_attr("a").unwrap();
    reads.lock().unwrap().clear();
    file.commit().unwrap();
    let requested = mem::take(&mut *reads.lock().unwrap());
    assert!(requested.iter().all(|read| !read.overlaps(numeric)));
    let mut scans = requested
        .into_iter()
        .filter(|read| read.overlaps(descriptors))
        .collect::<Vec<_>>();
    scans.sort_unstable_by_key(|read| read.start());
    let mut covered = descriptors.start();
    for read in &scans {
        assert!(read.length() <= 64 * 1024, "{read:?}");
        assert!(read.start() <= covered, "gap before {read:?}");
        covered = covered.max(read.end());
    }
    assert!(covered >= descriptors.end(), "{scans:?}");
    let total = scans.iter().map(|read| read.length()).sum::<u64>();
    assert!(total <= 4 * descriptors.length(), "{total}");
    assert!(large.attrs().unwrap().is_empty());
    drop(large);
    file.close().unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let parent = usize::try_from(parent).unwrap();
    let compacted = global_heap::collection(&[retained], Widths::EIGHT);
    // Compare the collection header, retained member, and free header.
    let prefix = 16 + 24 + 16;
    assert_eq!(bytes[parent..parent + prefix], compacted[..prefix]);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("labels").unwrap().read_string().unwrap(),
        vec!["retained"; 8_192]
    );
    let large = file.dataset("large").unwrap();
    assert!(large.attrs().unwrap().is_empty());
    assert_eq!(large.read_u8().unwrap(), vec![0x2a; LARGE]);
}

#[test]
fn heap_compaction_leaves_large_stationary_payloads_unread() {
    let path = test_util::temp::temp_path("stationary-heap-commit-reads.h5");
    let payload = vec![b's'; LARGE];
    let numeric = vec![0x2a; LARGE];
    let retained = Object {
        index: 91,
        reference_count: 3,
        data: &payload,
    };
    let obsolete = Object::new(7, b"gone");
    let original = global_heap::collection(&[retained, obsolete], Widths::EIGHT);
    let compacted = global_heap::collection_of_size(&[retained], Widths::EIGHT, original.len());
    let mut fixture = Fixture::new();
    let parent = fixture.image.append(&original);
    let address = u64::try_from(parent).unwrap();
    let datatype = global_heap_fixture::vlen_string();
    let messages = Form::CompactDataset.messages(
        &mut fixture,
        &datatype,
        &global_heap_fixture::reference(u32::try_from(LARGE).unwrap(), address, 91),
    );
    fixture.object("label", messages);
    let mut messages =
        Form::ContiguousDataset.messages(&mut fixture, &crate::make_u8_type(), &numeric);
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &datatype,
        &global_heap_fixture::reference(4, address, 7),
    ));
    fixture.object("large", messages);
    fixture.write(&path);

    // Eight-byte lengths give this collection and its records 16-byte headers.
    let payload_start = parent + 32;
    let free_header = payload_start + LARGE..payload_start + LARGE + 16;
    let member_payload = FileRange::new(
        PhysicalAddress::new(u64::try_from(payload_start).unwrap()),
        u64::try_from(LARGE).unwrap(),
    )
    .unwrap();
    let reads = Arc::new(Mutex::new(Vec::new()));
    let file = open_observed_file(&path, &reads);
    let numbers = extent(&file, "large");
    assert_eq!(numbers.length(), u64::try_from(LARGE).unwrap());
    let forbidden = [member_payload, numbers];

    file.dataset("large").unwrap().remove_attr("a").unwrap();
    drop(mem::take(&mut *reads.lock().unwrap()));
    let commit = || file.commit().unwrap();
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let ((), measured) = allocation::measure("stationary_heap_member_commit", commit);
        assert!(measured.peak_bytes < 64 * 1024, "{measured}");
        assert!(measured.live_bytes < 16 * 1024, "{measured}");
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    commit();
    let requested = mem::take(&mut *reads.lock().unwrap());
    for range in forbidden {
        assert!(
            requested.iter().all(|read| !read.overlaps(range)),
            "commit read payload {range:?}: {requested:?}"
        );
    }
    let bytes = requested.iter().map(|read| read.length()).sum::<u64>();
    assert!(
        (1..=16 * 1024).contains(&bytes),
        "commit requested {bytes} bytes"
    );
    assert!(!file.has_staged_edits());
    file.close().unwrap();

    let stored = std::fs::read(&path).unwrap();
    let prefix = free_header.start - parent;
    assert_eq!(&stored[parent..free_header.start], &original[..prefix]);
    assert_eq!(
        &stored[free_header.clone()],
        &compacted[prefix..prefix + 16]
    );
    assert_ne!(&stored[free_header], &original[prefix..prefix + 16]);

    let file = open_observed_file(&path, &reads);
    reads.lock().unwrap().clear();
    let read_values = || {
        (
            file.dataset("label").unwrap().read_string().unwrap(),
            file.dataset("large").unwrap().read_u8().unwrap(),
        )
    };
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let ((label, values), positive) =
        allocation::measure("stationary_heap_member_read_control", read_values);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let payload_bytes = u64::try_from(payload.len() + numeric.len()).unwrap();
        assert!(positive.peak_bytes >= payload_bytes, "{positive}");
        assert!(positive.live_bytes >= payload_bytes, "{positive}");
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let (label, values) = read_values();

    let control = mem::take(&mut *reads.lock().unwrap());
    for range in forbidden {
        assert!(control.iter().any(|read| read.overlaps(range)), "{range:?}");
    }
    let [label]: [String; 1] = label.try_into().unwrap();
    assert_eq!(label.as_bytes(), payload);
    assert_eq!(values, numeric);
    assert!(file.dataset("large").unwrap().attrs().unwrap().is_empty());
    file.close().unwrap();
}

fn open_observed_file(path: &Path, reads: &Arc<Mutex<Vec<FileRange>>>) -> File {
    File::from_test_edit_session(
        open_observed_engine(path, Arc::clone(reads)),
        FileAccessProperties::new()
            .with_metadata_cache(MetadataCacheConfig::disabled())
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap()
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
