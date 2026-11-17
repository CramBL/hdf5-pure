use hdf5_pure::{
    AttrValue, EditBacking, File, FileAccessProperties, FileBuilder, Layout, MemoryStrategy,
    SyncPolicy,
};
use rstest::rstest;
use test_util::allocation;

#[rstest]
#[case::group(Edit::Group)]
#[case::numeric_overwrite(Edit::Overwrite)]
#[case::numeric_attribute(Edit::Attribute)]
fn candidate_free_public_commits_have_bounded_allocations(#[case] edit: Edit) {
    let path = test_util::temp::temp_path("candidate-free-commit-allocations.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("value").with_i32_data(&[7]);
    builder.write(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(MemoryStrategy::Bounded)
            .with_page_buffer_size(0)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    match edit {
        Edit::Group => {
            file.root().create_group("added").unwrap();
        }
        Edit::Overwrite => file
            .dataset("value")
            .unwrap()
            .write_staged(|builder| {
                builder.with_i32_data(&[11]);
            })
            .unwrap(),
        Edit::Attribute => file
            .dataset("value")
            .unwrap()
            .set_attr("a", AttrValue::I64(9))
            .unwrap(),
    }
    let region = format!("candidate_free_public_commit_{edit:?}");
    let ((), measured) = allocation::measure(&region, || file.commit().unwrap());
    assert!(measured.blocks > 0 && measured.blocks <= 512, "{measured}");
    assert!(
        measured.bytes > 0 && measured.bytes <= 131_072,
        "{measured}"
    );
    assert_eq!(
        file.dataset("value").unwrap().read_i32().unwrap(),
        [if matches!(edit, Edit::Overwrite) {
            11
        } else {
            7
        }]
    );
    if matches!(edit, Edit::Group) {
        assert_eq!(file.root().groups().unwrap(), ["added"]);
    }
    if matches!(edit, Edit::Attribute) {
        assert_eq!(
            file.dataset("value").unwrap().attrs().unwrap()["a"],
            AttrValue::I64(9)
        );
    }
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("value").unwrap().read_i32().unwrap(),
        [if matches!(edit, Edit::Overwrite) {
            11
        } else {
            7
        }]
    );
    file.close().unwrap();
}

#[rstest]
#[case::allocated_one_mib(StorageAllocation::Allocated, 262_144)]
#[case::allocated_four_mib(StorageAllocation::Allocated, 1_048_576)]
#[case::unallocated_one_mib(StorageAllocation::Unallocated, 262_144)]
#[case::unallocated_four_mib(StorageAllocation::Unallocated, 1_048_576)]
fn staged_overwrite_commits_avoid_copying_large_replacement_buffers(
    #[case] storage: StorageAllocation,
    #[case] elements: usize,
) {
    let path = test_util::temp::temp_path("large-staged-overwrite-allocation.h5");
    let original = vec![7_i32; elements];
    let replacement: Vec<i32> = (0..elements)
        .map(|index| i32::try_from(index).unwrap())
        .collect();
    let shape = [u64::try_from(elements).unwrap()];
    let payload_bytes = u64::try_from(size_of_val(replacement.as_slice())).unwrap();
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("value");
    match storage {
        StorageAllocation::Allocated => dataset.with_i32_data(&original),
        StorageAllocation::Unallocated => {
            dataset.with_unallocated_storage(hdf5_pure::make_i32_type(), &shape)
        }
    };
    builder.write(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(MemoryStrategy::Bounded)
            .with_page_buffer_size(0)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    assert_eq!(file.edit_backing().unwrap(), EditBacking::Bounded);
    let Layout::Contiguous { address, size } = file.dataset("value").unwrap().layout().unwrap()
    else {
        panic!("expected contiguous dataset storage");
    };
    assert_eq!(
        address.is_some(),
        matches!(storage, StorageAllocation::Allocated)
    );
    assert_eq!(size, payload_bytes);
    assert_eq!(
        file.dataset("value").unwrap().read_i32().unwrap(),
        match storage {
            StorageAllocation::Allocated => original,
            StorageAllocation::Unallocated => vec![0; elements],
        }
    );
    file.dataset("value")
        .unwrap()
        .write_staged(|builder| {
            builder.with_i32_data(&replacement);
        })
        .unwrap();
    assert!(file.has_staged_edits());

    let region = format!("large_staged_overwrite_{storage:?}_{elements}");
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
    // An allocated overwrite reads one complete old payload into its undo journal. Unallocated
    // storage has no old bytes to journal. A replacement-buffer copy must not fit either bound.
    let journal_bytes = match storage {
        StorageAllocation::Allocated => payload_bytes,
        StorageAllocation::Unallocated => 0,
    };
    let overhead = measured.bytes.checked_sub(journal_bytes).unwrap();
    assert!(measured.peak_bytes >= journal_bytes, "{measured}");
    assert!(
        overhead < 65_536,
        "{measured}; undo journal: {journal_bytes}"
    );
    assert!(
        overhead < payload_bytes / 8,
        "{measured}; undo journal: {journal_bytes}"
    );
    assert!(!file.has_staged_edits());
    assert_eq!(file.dataset("value").unwrap().shape().unwrap(), shape);
    assert_eq!(
        file.dataset("value").unwrap().read_i32().unwrap(),
        replacement
    );
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("value").unwrap().shape().unwrap(), shape);
    assert_eq!(
        file.dataset("value").unwrap().read_i32().unwrap(),
        replacement
    );
    file.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum StorageAllocation {
    Allocated,
    Unallocated,
}

#[derive(Clone, Copy, Debug)]
enum Edit {
    Attribute,
    Group,
    Overwrite,
}
