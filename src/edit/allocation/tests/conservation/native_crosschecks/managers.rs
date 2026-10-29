//! Crosschecks manager rejection using files written by libhdf5.

use std::fs;

use rstest::rstest;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Interval;
use test_util::free_space_damage::Damage;
use test_util_hdf5::allocation_publication;

use super::super::Observation;
use super::super::WritePath;
use super::super::managers;

#[rstest]
#[case::header_signature(Damage::HeaderSignature)]
#[case::header_version(Damage::HeaderVersion)]
#[case::header_client(Damage::HeaderClient)]
#[case::header_bits(Damage::HeaderBits)]
#[case::header_checksum(Damage::HeaderChecksum)]
#[case::used_length(Damage::UsedLength)]
#[case::undefined_list(Damage::UndefinedList)]
#[case::list_address(Damage::ListAddress)]
#[case::allocated_length(Damage::AllocatedLength)]
#[case::list_signature(Damage::ListSignature)]
#[case::list_version(Damage::ListVersion)]
#[case::list_owner(Damage::ListOwner)]
#[case::list_checksum(Damage::ListChecksum)]
fn native_manager_damage_is_rejected_before_writes(
    #[case] damage: Damage,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
    #[values(WritePath::Immediate, WritePath::Borrowed, WritePath::Owned)] writer: WritePath,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-manager.h5");
    allocation_publication::write_fixture(&path, paged, deferred, false);
    let native = allocation_publication::Observation::read(&path);
    let pure = Observation::read(&fs::read(&path).unwrap());
    assert_eq!(native.values, pure.values);
    assert_eq!(native.chunks, pure.chunks);
    assert_eq!(
        Coverage::new(
            native
                .free
                .into_iter()
                .map(|(address, length)| Interval::new(address, length).unwrap())
        ),
        pure.free
    );
    managers::reject(&path, damage, writer);
}
