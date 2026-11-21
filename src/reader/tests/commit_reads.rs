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
    AttrValue, EditBacking, FileBuilder, FileSpaceSettings, FileSpaceStrategy, Layout,
    MetadataCacheConfig, PhysicalAddress, SyncPolicy,
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

#[derive(Debug, Clone, Copy)]
enum DescriptorPattern {
    Repeated,
    Distinct,
}

#[rstest]
#[case::repeated(DescriptorPattern::Repeated)]
#[case::distinct(DescriptorPattern::Distinct)]
fn heap_bearing_commit_bounds_descriptor_reads_and_skips_numeric_payload(
    #[case] pattern: DescriptorPattern,
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
) {
    let path = test_util::temp::temp_path("heap-bearing-commit-reads.h5");
    let (count, values) = match pattern {
        DescriptorPattern::Repeated => (8_192, vec!["retained".to_owned()]),
        DescriptorPattern::Distinct => (
            8_193,
            (0..8_193).map(|index| format!("{index:08}")).collect(),
        ),
    };
    let retained: Vec<_> = (91_u16..)
        .zip(&values)
        .map(|(index, value)| Object::new(index, value.as_bytes()))
        .collect();
    let obsolete = Object::new(7, b"gone");
    let original: Vec<_> = match pattern {
        DescriptorPattern::Repeated => core::iter::once(obsolete)
            .chain(retained.iter().copied())
            .collect(),
        DescriptorPattern::Distinct => retained.iter().copied().chain([obsolete]).collect(),
    };
    let original = global_heap::collection(&original, Widths::EIGHT);
    let compacted = global_heap::collection_of_size(&retained, Widths::EIGHT, original.len());
    let mut fixture = Fixture::new();
    let parent = fixture.image.append(&original);
    let address = u64::try_from(parent).unwrap();
    let datatype = global_heap_fixture::vlen_string();
    let labels: Vec<_> = retained
        .iter()
        .cycle()
        .take(count)
        .flat_map(|member| global_heap_fixture::reference(8, address, u32::from(member.index)))
        .collect();
    let messages = Form::ContiguousDataset.messages(&mut fixture, &datatype, &labels);
    fixture.object("labels", messages);
    let numeric = vec![0x2a; LARGE];
    let mut messages =
        Form::ContiguousDataset.messages(&mut fixture, &crate::make_u8_type(), &numeric);
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &datatype,
        &global_heap_fixture::reference(4, address, 7),
    ));
    fixture.object("large", messages);
    fixture.write(&path);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let input_bytes = std::fs::metadata(&path).unwrap().len();

    let read_capacity = 1_024;
    let reads = Arc::new(Mutex::new(Vec::with_capacity(read_capacity)));
    let file = open_observed_file_with_backing(&path, backing, &reads);
    assert_eq!(file.edit_backing(), Some(backing));
    let descriptors = extent(&file, "labels");
    let numbers = extent(&file, "large");
    assert_eq!(descriptors.length(), u64::try_from(labels.len()).unwrap());
    assert_eq!(numbers.length(), u64::try_from(numeric.len()).unwrap());
    let mut large = file.dataset("large").unwrap();
    large.remove_attr("a").unwrap();
    reads.lock().unwrap().clear();

    let commit = || file.commit().unwrap();
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let ((), measured) = allocation::measure(
        &format!("descriptor_commit_{pattern:?}_{backing:?}"),
        commit,
    );
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    commit();
    let requested = mem::take(&mut *reads.lock().unwrap());
    assert!(requested.len() <= read_capacity, "{}", requested.len());
    assert!(requested.iter().all(|read| !read.overlaps(numbers)));
    let mut scans: Vec<_> = requested
        .iter()
        .copied()
        .filter(|read| read.overlaps(descriptors))
        .collect();
    scans.sort_unstable_by_key(|read| read.start());
    let mut covered = descriptors.start();
    for read in &scans {
        assert!(descriptors.contains(*read), "{read:?}");
        assert!(read.length() <= 64 * 1024, "{read:?}");
        assert!(read.start() <= covered, "gap before {read:?}");
        covered = covered.max(read.end());
    }
    assert_eq!(covered, descriptors.end(), "{scans:?}");
    let scan_bytes = scans.iter().map(|read| read.length()).sum::<u64>();
    assert_eq!(scan_bytes, descriptors.length(), "{scans:?}");
    let requested_bytes = requested.iter().map(|read| read.length()).sum::<u64>();
    assert!(
        requested_bytes <= descriptors.length() + 2 * original.len() as u64 + 16 * 1024,
        "{requested_bytes}"
    );
    let largest_read = requested.iter().map(|read| read.length()).max().unwrap();
    assert!(largest_read <= 64 * 1024, "{largest_read}");
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        // The mirror can grow when the commit places the new root past the input extent.
        let image_growth = match backing {
            EditBacking::Bounded => 0,
            EditBacking::Mirrored => 2 * input_bytes,
        };
        let (cumulative_limit, peak_limit, block_limit) = match pattern {
            DescriptorPattern::Repeated => (160 * 1024, 96 * 1024, 448),
            DescriptorPattern::Distinct => (2_816 * 1024, 1_664 * 1024, 8_000),
        };
        assert!(
            measured.bytes <= image_growth + cumulative_limit,
            "{measured}"
        );
        assert!(
            measured.peak_bytes <= image_growth + peak_limit,
            "{measured}"
        );
        assert!(measured.live_bytes <= image_growth + 8 * 1024, "{measured}");
        assert!((1..=block_limit).contains(&measured.blocks), "{measured}");
    }
    eprintln!(
        "{pattern:?} {backing:?}: {} requests, {requested_bytes} bytes, largest {largest_read}; {} descriptor requests, {scan_bytes} bytes",
        requested.len(),
        scans.len()
    );

    assert!(!file.has_staged_edits());
    assert!(large.attrs().unwrap().is_empty());
    drop(large);
    file.close().unwrap();

    let bytes = std::fs::read(&path).unwrap();
    // The fixture has an eight-byte payload in every retained record.
    let prefix = 16 + 24 * retained.len() + 16;
    assert_eq!(bytes[parent..parent + prefix], compacted[..prefix]);
    assert_ne!(bytes[parent..parent + prefix], original[..prefix]);
    let file = open_observed_file(&path, &reads);
    let expected: Vec<_> = values
        .iter()
        .cycle()
        .take(count)
        .map(String::as_str)
        .collect();
    assert_eq!(
        file.dataset("labels").unwrap().read_string().unwrap(),
        expected
    );
    let large = file.dataset("large").unwrap();
    assert!(large.attrs().unwrap().is_empty());
    reads.lock().unwrap().clear();
    let read_values = || large.read_u8().unwrap();
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let (result, positive) = allocation::measure(
        &format!("descriptor_control_{pattern:?}_{backing:?}"),
        read_values,
    );
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let result = read_values();
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    assert!(positive.live_bytes >= numeric.len() as u64, "{positive}");
    assert_eq!(result, numeric);
    assert!(
        reads
            .lock()
            .unwrap()
            .iter()
            .any(|read| read.overlaps(numbers))
    );
}

#[derive(Clone, Copy)]
enum DescriptorOverwrite {
    InPlace,
    Reconstructed,
}

#[rstest]
#[case::in_place(DescriptorOverwrite::InPlace)]
#[case::reconstructed(DescriptorOverwrite::Reconstructed)]
fn descriptor_overwrite_observes_new_heap_members(
    #[case] overwrite: DescriptorOverwrite,
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
) {
    let path = test_util::temp::temp_path("heap-bearing-overwrite.h5");
    let mut fixture = Fixture::new();
    let first = Object::new(91, b"original");
    let next = Object::new(7, b"retained");
    let parent = fixture.collection(&[first, next]);
    let datatype = global_heap_fixture::vlen_string();
    let (form, count) = match overwrite {
        DescriptorOverwrite::InPlace => (Form::ContiguousDataset, 4_097),
        DescriptorOverwrite::Reconstructed => (Form::CompactDataset, 1),
    };
    let before = global_heap_fixture::reference(8, parent, 91).repeat(count);
    let after = global_heap_fixture::reference(8, parent, 7).repeat(count);
    let messages = form.messages(&mut fixture, &datatype, &before);
    fixture.object("labels", messages);
    let mut messages = Form::CompactDataset.messages(&mut fixture, &crate::make_u8_type(), &[0x2a]);
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &datatype,
        &global_heap_fixture::reference(8, parent, 7),
    ));
    fixture.object("keeper", messages);
    fixture.write(&path);

    let reads = Arc::new(Mutex::new(Vec::new()));
    let file = open_observed_file_with_backing(&path, backing, &reads);
    assert_eq!(file.edit_backing(), Some(backing));
    let mut labels = file.dataset("labels").unwrap();
    let header = labels.header_address().unwrap();
    let layout = labels.layout().unwrap();
    let old_range = match overwrite {
        DescriptorOverwrite::InPlace => {
            let range = extent(&file, "labels");
            assert_eq!(range.length(), after.len() as u64);
            Some(range)
        }
        DescriptorOverwrite::Reconstructed => None,
    };
    assert_eq!(labels.read_raw().unwrap(), before);
    let mut keeper = file.dataset("keeper").unwrap();
    keeper.remove_attr("a").unwrap();
    labels
        .write_staged(|builder| {
            builder.with_raw_data(datatype.clone(), after.clone(), count as u64);
        })
        .unwrap();
    reads.lock().unwrap().clear();
    file.commit().unwrap();

    assert!(!file.has_staged_edits());
    let placed_header = labels.header_address().unwrap();
    let placed_layout = labels.layout().unwrap();
    if let Some(range) = old_range {
        assert_eq!(placed_header, header);
        assert_eq!(placed_layout, layout);
        // The journal reads the entire range between the two batched scans.
        let full = FileRange::new(range.start(), 64 * 1024).unwrap();
        let tail = FileRange::new(full.end(), 16).unwrap();
        let observed: Vec<_> = reads
            .lock()
            .unwrap()
            .iter()
            .copied()
            .filter(|read| read.overlaps(range))
            .collect();
        assert_eq!(observed, [full, tail, range, full, tail]);
    } else {
        assert_ne!(placed_header, header);
        let witness = FileRange::new(PhysicalAddress::new(placed_header), 4).unwrap();
        assert!(
            reads
                .lock()
                .unwrap()
                .iter()
                .any(|read| read.contains(witness))
        );
    }
    assert!(keeper.attrs().unwrap().is_empty());
    drop((labels, keeper));
    file.close().unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let parent = usize::try_from(parent).unwrap();
    let original = global_heap::collection(&[first, next], Widths::EIGHT);
    let compacted = global_heap::collection(&[next], Widths::EIGHT);
    // One retained eight-byte record, followed by the free header.
    let prefix = 16 + 24 + 16;
    assert_eq!(bytes[parent..parent + prefix], compacted[..prefix]);
    assert_ne!(bytes[parent..parent + prefix], original[..prefix]);
    let file = File::open(&path).unwrap();
    let labels = file.dataset("labels").unwrap();
    assert_eq!(labels.header_address().unwrap(), placed_header);
    assert_eq!(labels.layout().unwrap(), placed_layout);
    assert_eq!(labels.read_raw().unwrap(), after);
    assert_eq!(labels.read_string().unwrap(), vec!["retained"; count]);
    let keeper = file.dataset("keeper").unwrap();
    assert_eq!(keeper.read_u8().unwrap(), [0x2a]);
    assert!(keeper.attrs().unwrap().is_empty());
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
    open_observed_file_with_backing(path, EditBacking::Bounded, reads)
}

fn open_observed_file_with_backing(
    path: &Path,
    backing: EditBacking,
    reads: &Arc<Mutex<Vec<FileRange>>>,
) -> File {
    File::from_test_edit_session(
        open_observed_engine(path, backing, Arc::clone(reads)),
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
