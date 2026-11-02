#![cfg(feature = "__hdf5-1.10")]

use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::SyncPolicy;
use rstest::rstest;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Interval;
use test_util_hdf5::file_space;
use test_util_hdf5::file_space::admission::Chain;
use test_util_hdf5::file_space::admission::Fixture;
use test_util_hdf5::file_space::admission::Origin;
use test_util_hdf5::file_space::admission::Scale;
use test_util_hdf5::file_space::admission::Threshold;
use test_util_hdf5::file_space_oracle;

#[rstest]
#[case::native_flat(Origin::Native, Chain::Flat { scale: Scale::TwoThousand, threshold: Threshold::Allocation })]
#[case::pure_flat(Origin::Pure, Chain::Flat { scale: Scale::TwoThousand, threshold: Threshold::Allocation })]
#[case::native_page(Origin::Native, Chain::Page)]
#[case::pure_page(Origin::Pure, Chain::Page)]
fn seeded_chunk_deletion_preserves_publication_reopen_and_reuse(
    #[case] origin: Origin,
    #[case] chain: Chain,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("admission.h5");
    let result = Fixture::create(&path, origin, chain).commit(&path, memory);
    assert!(!result.free.is_empty());
    assert!(result.sections > 0);
    assert!(result.manager_bytes > 0);
    let values = vec![19_u8; 65_536];
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(memory)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    file.root()
        .create_dataset("replacement", |builder| {
            builder.with_u8_data(&values);
        })
        .unwrap();
    file.commit().unwrap();
    let allocation = file_space::contiguous_extent(&file, "replacement");
    let allocated =
        Coverage::new([
            Interval::new(allocation.start, allocation.end - allocation.start).unwrap(),
        ]);
    assert!(allocated.difference(&result.free).is_empty());
    assert_eq!(
        file.dataset("replacement").unwrap().read_u8().unwrap(),
        values
    );
    file.close().unwrap();
    let native = hdf5::File::open(&path).unwrap();
    assert_eq!(
        native
            .dataset("replacement")
            .unwrap()
            .read_raw::<u8>()
            .unwrap(),
        values
    );
    assert_eq!(
        native.dataset("keep").unwrap().read_raw::<u32>().unwrap(),
        vec![77; 4_096]
    );
    native.close().unwrap();
    file_space_oracle::assert_persisted_sections(&path);
}
