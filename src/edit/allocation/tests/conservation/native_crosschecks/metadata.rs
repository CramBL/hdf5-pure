//! Checks append rejection after damaging index metadata in libhdf5-produced files.

use std::fs;

use rstest::rstest;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Interval;
use test_util_hdf5::allocation_publication;

use super::super::Observation;
use super::super::WritePath;
use super::super::metadata;
use super::super::metadata::Damage;

#[rstest]
#[case::header_checksum(Damage::HeaderChecksum)]
#[case::index_checksum(Damage::IndexChecksum)]
#[case::index_signature(Damage::IndexSignature)]
#[case::index_owner(Damage::IndexOwner)]
#[case::direct_checksum(Damage::DirectChecksum)]
#[case::direct_signature(Damage::DirectSignature)]
fn native_indexes_reject_edits_after_framing_or_checksum_damage(
    #[case] damage: Damage,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
    #[values(false, true)] filtered: bool,
    #[values(WritePath::Immediate, WritePath::Borrowed, WritePath::Owned)] writer: WritePath,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-index.h5");
    allocation_publication::write_fixture(&path, paged, deferred, filtered);
    let before = allocation_publication::Observation::read(&path);
    assert_eq!(
        before.values,
        Observation::read(&fs::read(&path).unwrap()).values
    );
    metadata::initialize(&path);
    let native = allocation_publication::Observation::read(&path);
    let pure = Observation::read(&fs::read(&path).unwrap());
    assert_eq!(native.values, pure.values);
    assert_eq!(native.chunks, pure.chunks);
    assert_eq!(
        Coverage::new(
            native
                .free
                .into_iter()
                .map(|(address, length)| { Interval::new(address, length).unwrap() })
        ),
        pure.free
    );
    metadata::reject(&path, damage, writer);
}
