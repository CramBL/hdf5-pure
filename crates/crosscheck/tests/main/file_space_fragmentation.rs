#![cfg(feature = "__hdf5-1.10")]

use hdf5::File as NativeFile;
use hdf5_pure::AttrValue;
use hdf5_pure::File;
use hdf5_pure::PersistedFreeSpacePolicy;
use rstest::rstest;
use test_util_hdf5::file_space::fragmentation;
use test_util_hdf5::file_space::fragmentation::Fragmentation;
use test_util_hdf5::file_space_oracle;

#[rstest]
#[case::thousand(Fragmentation::Thousand)]
#[case::two_thousand(Fragmentation::TwoThousand)]
#[case::four_thousand(Fragmentation::FourThousand)]
#[case::eight_thousand(Fragmentation::EightThousand)]
fn native_fragmentation_preserves_managers_survivors_and_publication(
    #[case] scale: Fragmentation,
    #[values(PersistedFreeSpacePolicy::Tolerant, PersistedFreeSpacePolicy::Strict)]
    policy: PersistedFreeSpacePolicy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fragmented.h5");
    scale.write_native(&path);
    let before = fragmentation::persisted(&path);
    let size = std::fs::metadata(&path).unwrap().len();
    for _ in 0..3 {
        let file =
            File::open_rw_with_options(&path, fragmentation::bounded_options(policy)).unwrap();
        assert!(
            file.reusable_space_stats().unwrap().reusable_free_bytes() >= scale.raw_free_bytes()
        );
        file.close().unwrap();
        assert_eq!(fragmentation::persisted(&path), before);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), size);
    }
    scale.assert_native_survivors(&path);
    let replacement = vec![0xA5_u32; fragmentation::PAGE_ELEMENTS];
    let file = File::open_rw_with_options(&path, fragmentation::bounded_options(policy)).unwrap();
    file.root()
        .create_dataset("replacement", |builder| {
            builder.with_u32_data(&replacement);
        })
        .unwrap();
    file.root()
        .set_attr("fragmentation_probe", AttrValue::U32(7))
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let native = NativeFile::open_rw(&path).unwrap();
    assert_eq!(
        native
            .dataset("replacement")
            .unwrap()
            .read_raw::<u32>()
            .unwrap(),
        replacement
    );
    assert_eq!(
        native
            .attr("fragmentation_probe")
            .unwrap()
            .read_scalar::<u32>()
            .unwrap(),
        7
    );
    native.flush().unwrap();
    native.close().unwrap();
    scale.assert_native_survivors(&path);
    file_space_oracle::assert_persisted_sections(&path);
    let file = File::open_rw_with_options(&path, fragmentation::bounded_options(policy)).unwrap();
    assert_eq!(
        file.dataset("replacement").unwrap().read_u32().unwrap(),
        replacement
    );
    file.root().delete("replacement").unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    scale.assert_native_survivors(&path);
    let native = NativeFile::open_rw(&path).unwrap();
    assert!(!native.link_exists("replacement"));
    native.flush().unwrap();
    native.close().unwrap();
    file_space_oracle::assert_persisted_sections(&path);
}
