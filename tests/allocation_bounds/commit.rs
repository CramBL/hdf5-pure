use std::path::Path;

use hdf5_pure::{
    AttrValue, EditBacking, File, FileAccessProperties, FileBuilder, Layout, MemoryStrategy,
    SyncPolicy,
};
use rstest::rstest;
use test_util::allocation;

#[rstest]
#[case::group(Edit::Group)]
#[case::numeric_attribute(Edit::Attribute)]
#[case::numeric_overwrite(Edit::Overwrite)]
fn candidate_free_public_commits_have_bounded_allocations(#[case] edit: Edit) {
    let path = test_util::temp::temp_path("candidate-free-commit-allocations.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("value").with_i32_data(&[7]);
    builder.write(&path).unwrap();
    let file = open_editor(&path);
    match edit {
        Edit::Group => {
            file.root().create_group("added").unwrap();
        }
        Edit::Attribute => file
            .dataset("value")
            .unwrap()
            .set_attr("a", AttrValue::I64(9))
            .unwrap(),
        Edit::Overwrite => stage_overwrite(&file, &[11]),
    }
    assert!(file.has_staged_edits());

    let region = format!("candidate_free_public_commit_{edit:?}");
    let ((), measured) = allocation::measure(&region, || file.commit().unwrap());
    assert!((1..=512).contains(&measured.blocks), "{measured}");
    assert!((1..=128 * 1024).contains(&measured.bytes), "{measured}");
    assert!(!file.has_staged_edits());
    edit.assert_values(&file);
    file.close().unwrap();
    let reopened = File::open(&path).unwrap();
    edit.assert_values(&reopened);
    reopened.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Edit {
    Group,
    Attribute,
    Overwrite,
}

impl Edit {
    fn assert_values(self, file: &File) {
        let dataset = file.dataset("value").unwrap();
        let expected = if matches!(self, Self::Overwrite) {
            11
        } else {
            7
        };
        assert_eq!(dataset.read_i32().unwrap(), [expected]);
        let attrs = dataset.attrs().unwrap();
        if matches!(self, Self::Attribute) {
            assert_eq!(attrs.len(), 1);
            assert_eq!(attrs["a"], AttrValue::I64(9));
        } else {
            assert!(attrs.is_empty());
        }
        let groups = file.root().groups().unwrap();
        if matches!(self, Self::Group) {
            assert_eq!(groups, ["added"]);
        } else {
            assert!(groups.is_empty());
        }
    }
}

#[rstest]
#[case::allocated_one_mib(Storage::Allocated, 1)]
#[case::allocated_four_mib(Storage::Allocated, 4)]
#[case::unallocated_one_mib(Storage::Unallocated, 1)]
#[case::unallocated_four_mib(Storage::Unallocated, 4)]
fn staged_overwrite_commits_avoid_copying_large_replacement_buffers(
    #[case] storage: Storage,
    #[case] mebibytes: usize,
) {
    let path = test_util::temp::temp_path("large-staged-overwrite-allocation.h5");
    let elements = (mebibytes << 20) / size_of::<i32>();
    let shape = [u64::try_from(elements).unwrap()];
    let replacement: Vec<i32> = (0..i32::try_from(elements).unwrap()).collect();
    let payload_bytes = u64::try_from(size_of_val(replacement.as_slice())).unwrap();
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("value");
    match storage {
        Storage::Allocated => dataset.with_i32_data(&vec![7; elements]),
        Storage::Unallocated => {
            dataset.with_unallocated_storage(hdf5_pure::make_i32_type(), &shape)
        }
    };
    builder.write(&path).unwrap();

    let file = open_editor(&path);
    let (address, size) = contiguous_layout(&file);
    assert_eq!(address.is_some(), matches!(storage, Storage::Allocated));
    assert_eq!(size, payload_bytes);
    let fill = if matches!(storage, Storage::Allocated) {
        7
    } else {
        0
    };
    assert_eq!(
        file.dataset("value").unwrap().read_i32().unwrap(),
        vec![fill; elements]
    );
    stage_overwrite(&file, &replacement);
    assert!(file.has_staged_edits());

    let region = format!("large_staged_overwrite_{storage:?}_{mebibytes}_mib");
    let (copy, positive) = allocation::measure(&format!("{region}_copy"), || {
        std::hint::black_box(replacement.clone())
    });
    assert_eq!(copy, replacement);
    assert!(positive.blocks > 0, "{positive}");
    assert!(positive.bytes >= payload_bytes, "{positive}");
    assert!(positive.live_bytes >= payload_bytes, "{positive}");
    drop(copy);

    let ((), measured) = allocation::measure(&region, || file.commit().unwrap());
    assert!(measured.blocks > 0 && measured.bytes > 0, "{measured}");
    // In-place overwrite journals one complete old payload. Unallocated storage has no old
    // payload to journal. The remaining allowance covers commit metadata.
    let journal = if address.is_some() { payload_bytes } else { 0 };
    assert!(
        measured.bytes >= journal,
        "{measured}; undo journal: {journal}"
    );
    let overhead = measured.bytes - journal;
    assert!(measured.peak_bytes >= journal, "{measured}");
    assert!(measured.live_bytes < 64 * 1024, "{measured}");
    assert!(overhead < 64 * 1024, "{measured}; undo journal: {journal}");
    assert!(
        overhead < payload_bytes / 8,
        "{measured}; undo journal: {journal}"
    );
    assert!(!file.has_staged_edits());

    let (committed_address, committed_size) = contiguous_layout(&file);
    assert!(committed_address.is_some());
    if let Some(address) = address {
        assert_eq!(committed_address, Some(address));
    }
    assert_eq!(committed_size, payload_bytes);
    assert_overwritten(&file, shape, &replacement);
    file.close().unwrap();

    let reopened = File::open(&path).unwrap();
    assert_eq!(
        contiguous_layout(&reopened),
        (committed_address, payload_bytes)
    );
    assert_overwritten(&reopened, shape, &replacement);
    reopened.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Storage {
    Allocated,
    Unallocated,
}

fn open_editor(path: &Path) -> File {
    let file = File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_memory_strategy(MemoryStrategy::Bounded)
            .with_page_buffer_size(0)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    assert_eq!(file.edit_backing(), Some(EditBacking::Bounded));
    file
}

fn stage_overwrite(file: &File, values: &[i32]) {
    file.dataset("value")
        .unwrap()
        .write_staged(|builder| {
            builder.with_i32_data(values);
        })
        .unwrap();
}

fn contiguous_layout(file: &File) -> (Option<u64>, u64) {
    let Layout::Contiguous { address, size } = file.dataset("value").unwrap().layout().unwrap()
    else {
        panic!("expected contiguous value storage");
    };
    (address, size)
}

fn assert_overwritten(file: &File, shape: [u64; 1], values: &[i32]) {
    let dataset = file.dataset("value").unwrap();
    assert_eq!(dataset.shape().unwrap(), shape);
    assert_eq!(dataset.read_i32().unwrap(), values);
}
