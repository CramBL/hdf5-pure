use alloc::vec;
use alloc::vec::Vec;

use rstest::rstest;

use super::support;
use crate::address::{StoredAddress, StoredAddressExt};
use crate::error::FormatError;
use crate::link::creation_order::{LinkCreationCounter, LinkCreationPolicy};
use crate::link::info::{LinkCreationOrderIndex, LinkInfoMessage};
use crate::link::storage::{GroupLinkStorage, LinkStorageError, OwnedLink, OwnedLinkTarget};
use crate::width::OffsetWidth;

#[rstest]
#[case::untracked(LinkCreationPolicy::Untracked)]
#[case::tracked(LinkCreationPolicy::Tracked { next: counter(11) })]
#[case::indexed(LinkCreationPolicy::Indexed { next: counter(11) })]
fn link_info_round_trips_supported_policy_and_storage_combinations(
    #[case] policy: LinkCreationPolicy,
    #[values(false, true)] dense: bool,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let original = info_at(width, policy, dense);
    let links = vec![support::hard_link(
        b"a".to_vec(),
        policy.next_index().map(|_| 3),
    )];
    let bytes = original.to_bytes();
    let storage = GroupLinkStorage::parse(&bytes, links.clone(), width).unwrap();
    assert_eq!(storage.info(), original);
    assert_eq!(storage.links(), links);
    assert_eq!(storage.creation_order(), policy);
}

#[rstest]
#[case::untracked(LinkCreationPolicy::Untracked, None)]
#[case::tracked(LinkCreationPolicy::Tracked { next: counter(11) }, Some(11))]
#[case::indexed(LinkCreationPolicy::Indexed { next: counter(11) }, Some(11))]
fn deletions_preserve_gaps_and_new_links_use_the_stored_counter(
    #[case] policy: LinkCreationPolicy,
    #[case] assigned: Option<u64>,
) {
    let indexes = policy.next_index().map(|_| (2, 7));
    let mut storage = GroupLinkStorage::new(
        &info(policy, true),
        vec![
            support::hard_link(b"a".to_vec(), indexes.map(|x| x.0)),
            support::hard_link(b"b".to_vec(), indexes.map(|x| x.1)),
        ],
    )
    .unwrap();
    assert_eq!(
        storage.remove(b"b").unwrap().creation_order(),
        indexes.map(|x| x.1)
    );
    assert!(storage.remove(b"missing").is_none());
    storage
        .insert(support::hard_link(b"c".to_vec(), None))
        .unwrap();
    assert_eq!(storage.links()[0].creation_order(), indexes.map(|x| x.0));
    assert_eq!(storage.links()[1].creation_order(), assigned);
    assert_eq!(
        storage.creation_order().next_index(),
        policy.next_index().map(|index| index + 1)
    );
    storage.use_compact_storage();
    assert_eq!(
        storage.info().creation_order_index(),
        if policy.is_indexed() {
            LinkCreationOrderIndex::Undefined
        } else {
            LinkCreationOrderIndex::Absent
        }
    );
    let compact = storage.info().to_bytes();
    let reparsed =
        GroupLinkStorage::parse(&compact, storage.links().to_vec(), OffsetWidth::Eight).unwrap();
    assert_eq!(storage, reparsed);
    storage
        .use_dense_storage(
            StoredAddress::new(512),
            StoredAddress::new(1024),
            policy.is_indexed().then(|| StoredAddress::new(2048)),
        )
        .unwrap();
    assert_eq!(
        storage.info().max_creation_order(),
        policy.next_index().map(|index| index + 1)
    );
}

#[rstest]
#[case::tracked(LinkCreationPolicy::Tracked { next: counter(i64::MAX.cast_unsigned()) })]
#[case::indexed(LinkCreationPolicy::Indexed { next: counter(i64::MAX.cast_unsigned()) })]
fn an_exhausted_counter_rejects_assignment_without_mutation(
    #[case] mut policy: LinkCreationPolicy,
) {
    let original = policy;
    assert_eq!(
        policy.take().unwrap_err(),
        LinkStorageError::CreationOrderExhausted
    );
    assert_eq!(policy, original);
    let mut storage = GroupLinkStorage::new(&info(policy, false), vec![]).unwrap();
    let original = storage.clone();
    assert_eq!(
        storage
            .insert(support::hard_link(b"a".to_vec(), None))
            .unwrap_err(),
        LinkStorageError::CreationOrderExhausted
    );
    assert_eq!(storage, original);
}

#[rstest]
#[case::tracked(LinkCreationPolicy::Tracked { next: counter(i64::MAX.cast_unsigned() - 1) })]
#[case::indexed(LinkCreationPolicy::Indexed { next: counter(i64::MAX.cast_unsigned() - 1) })]
fn the_last_incrementable_counter_is_assigned(#[case] mut policy: LinkCreationPolicy) {
    assert_eq!(policy.take(), Ok(Some(i64::MAX.cast_unsigned() - 1)));
    assert_eq!(policy.next_index(), Some(i64::MAX.cast_unsigned()));
    assert_eq!(policy.take(), Err(LinkStorageError::CreationOrderExhausted));
}

#[rstest]
#[case::sign_bit(i64::MAX.cast_unsigned() + 1)]
#[case::unsigned_last(u64::MAX - 1)]
#[case::unsigned_max(u64::MAX)]
fn out_of_range_counters_cannot_construct_writer_policy_or_storage(
    #[case] next: u64,
    #[values(false, true)] indexed: bool,
) {
    let error = LinkStorageError::CreationOrderCounterRange {
        next,
        maximum: i64::MAX.cast_unsigned(),
    };
    assert_eq!(LinkCreationCounter::new(next).unwrap_err(), error);
    let message = LinkInfoMessage::compact(OffsetWidth::Eight, Some(next), indexed);
    assert_eq!(LinkCreationPolicy::from_info(&message).unwrap_err(), error);
    assert_eq!(
        GroupLinkStorage::new(&message, vec![support::hard_link(b"a".to_vec(), Some(0))])
            .unwrap_err(),
        error
    );
}

#[rstest]
#[case::missing_index(LinkCreationPolicy::Tracked { next: counter(5) }, vec![support::hard_link(b"a".to_vec(), None)], LinkStorageError::CreationOrderPresence)]
#[case::unexpected_index(LinkCreationPolicy::Untracked, vec![support::hard_link(b"a".to_vec(), Some(0))], LinkStorageError::CreationOrderPresence)]
#[case::counter_at_index(LinkCreationPolicy::Tracked { next: counter(5) }, vec![support::hard_link(b"a".to_vec(), Some(5))], LinkStorageError::CreationOrderCounter { index: 5, next: 5 })]
#[case::counter_below_index(LinkCreationPolicy::Tracked { next: counter(5) }, vec![support::hard_link(b"a".to_vec(), Some(7))], LinkStorageError::CreationOrderCounter { index: 7, next: 5 })]
#[case::duplicate_index(LinkCreationPolicy::Tracked { next: counter(5) }, vec![support::hard_link(b"a".to_vec(), Some(0)), support::hard_link(b"b".to_vec(), Some(0))], LinkStorageError::DuplicateCreationOrder(0))]
#[case::duplicate_name(LinkCreationPolicy::Untracked, vec![support::hard_link(b"a".to_vec(), None), support::hard_link(b"a".to_vec(), None)], LinkStorageError::DuplicateName(b"a".to_vec()))]
fn inconsistent_link_sets_are_rejected(
    #[case] policy: LinkCreationPolicy,
    #[case] links: Vec<OwnedLink>,
    #[case] error: LinkStorageError,
) {
    assert_eq!(
        GroupLinkStorage::new(&info(policy, false), links).unwrap_err(),
        error
    );
}

#[rstest]
#[case::heap_only(LinkInfoMessage::new(OffsetWidth::Eight, None, Some(StoredAddress::new(512)), None, LinkCreationOrderIndex::Absent).unwrap(), LinkStorageError::StorageAddresses)]
#[case::name_only(LinkInfoMessage::new(OffsetWidth::Eight, None, None, Some(StoredAddress::new(512)), LinkCreationOrderIndex::Absent).unwrap(), LinkStorageError::StorageAddresses)]
#[case::index_without_tracking(
    LinkInfoMessage::compact(OffsetWidth::Eight, None, true),
    LinkStorageError::IndexingWithoutTracking
)]
#[case::missing_dense_index(LinkInfoMessage::new(OffsetWidth::Eight, Some(0), Some(StoredAddress::new(512)), Some(StoredAddress::new(1024)), LinkCreationOrderIndex::Undefined).unwrap(), LinkStorageError::StorageAddresses)]
#[case::compact_with_index(LinkInfoMessage::new(OffsetWidth::Eight, Some(0), None, None, LinkCreationOrderIndex::Defined(StoredAddress::new(512))).unwrap(), LinkStorageError::StorageAddresses)]
fn inconsistent_storage_addresses_are_rejected(
    #[case] message: LinkInfoMessage,
    #[case] error: LinkStorageError,
) {
    assert_eq!(GroupLinkStorage::new(&message, vec![]).unwrap_err(), error);
}

#[test]
fn invalid_dense_replacement_preserves_the_existing_representation() {
    let mut storage = GroupLinkStorage::new(
        &info_at(
            OffsetWidth::Two,
            LinkCreationPolicy::Indexed { next: counter(0) },
            false,
        ),
        vec![],
    )
    .unwrap();
    let original = storage.clone();
    assert_eq!(
        storage
            .use_dense_storage(StoredAddress::new(512), StoredAddress::new(1024), None,)
            .unwrap_err(),
        LinkStorageError::StorageAddresses
    );
    assert_eq!(storage, original);
    assert_eq!(
        storage
            .use_dense_storage(
                StoredAddress::new(0xffff),
                StoredAddress::new(1024),
                Some(StoredAddress::new(2048)),
            )
            .unwrap_err(),
        LinkStorageError::UndefinedAddress
    );
    assert_eq!(storage, original);
    assert_eq!(
        storage
            .use_dense_storage(
                StoredAddress::new(u64::MAX),
                StoredAddress::new(1024),
                Some(StoredAddress::new(2048)),
            )
            .unwrap_err(),
        LinkStorageError::Format(FormatError::AddressTooLarge {
            address: u64::MAX,
            offset_size: 2
        })
    );
    assert_eq!(storage, original);
}

#[test]
fn duplicate_or_preindexed_insertions_preserve_the_link_set_and_counter() {
    let mut storage = GroupLinkStorage::new(
        &info(LinkCreationPolicy::Indexed { next: counter(1) }, false),
        vec![support::hard_link(b"a".to_vec(), Some(0))],
    )
    .unwrap();
    let original = storage.clone();
    assert_eq!(
        storage
            .insert(support::hard_link(b"a".to_vec(), None))
            .unwrap_err(),
        LinkStorageError::DuplicateName(b"a".to_vec())
    );
    assert_eq!(storage, original);
    assert_eq!(
        storage
            .insert(support::hard_link(b"b".to_vec(), Some(8)))
            .unwrap_err(),
        LinkStorageError::CreationOrderPresence
    );
    assert_eq!(storage, original);
}

#[test]
fn group_retargeting_preserves_names_creation_indexes_and_storage_policy() {
    let original_info = info(LinkCreationPolicy::Indexed { next: counter(12) }, true);
    let mut storage = GroupLinkStorage::new(
        &original_info,
        vec![support::hard_link(b"a".to_vec(), Some(3))],
    )
    .unwrap();
    assert!(
        !storage
            .set_hard_target(b"missing", StoredAddress::new(4096))
            .unwrap()
    );
    assert!(
        storage
            .set_hard_target(b"a", StoredAddress::new(4096))
            .unwrap()
    );
    assert_eq!(storage.info(), original_info);
    assert_eq!(storage.links()[0].name(), b"a");
    assert_eq!(storage.links()[0].creation_order(), Some(3));
    assert_eq!(
        storage.links()[0].target(),
        &OwnedLinkTarget::Hard {
            object_header_address: StoredAddress::new(4096)
        }
    );
}

#[test]
fn group_encoding_width_rejects_oversized_targets_before_insertion() {
    let mut link = support::hard_link(b"a".to_vec(), None);
    link.set_hard_target(StoredAddress::new(0x1_0000), OffsetWidth::Eight)
        .unwrap();
    let message = LinkInfoMessage::compact(OffsetWidth::Two, None, false);
    let error = LinkStorageError::Format(FormatError::AddressTooLarge {
        address: 0x1_0000,
        offset_size: 2,
    });
    assert_eq!(
        GroupLinkStorage::new(&message, vec![link.clone()]).unwrap_err(),
        error
    );
    let mut storage = GroupLinkStorage::new(&message, vec![]).unwrap();
    assert_eq!(storage.offset_width(), OffsetWidth::Two);
    assert_eq!(storage.insert(link).unwrap_err(), error);
    assert!(storage.links().is_empty());
}

#[test]
fn reserved_link_info_flags_and_trailing_bytes_are_rejected() {
    let mut bytes = info(LinkCreationPolicy::Untracked, false).to_bytes();
    bytes[1] = 4;
    assert_eq!(
        GroupLinkStorage::parse(&bytes, vec![], OffsetWidth::Eight).unwrap_err(),
        LinkStorageError::Flags(4)
    );
    bytes[1] = 0;
    bytes.push(0);
    assert_eq!(
        GroupLinkStorage::parse(&bytes, vec![], OffsetWidth::Eight).unwrap_err(),
        LinkStorageError::TrailingBytes
    );
}

fn counter(next: u64) -> LinkCreationCounter {
    LinkCreationCounter::new(next).unwrap()
}

fn info(policy: LinkCreationPolicy, dense: bool) -> LinkInfoMessage {
    info_at(OffsetWidth::Eight, policy, dense)
}

fn info_at(width: OffsetWidth, policy: LinkCreationPolicy, dense: bool) -> LinkInfoMessage {
    if !dense {
        return LinkInfoMessage::compact(width, policy.next_index(), policy.is_indexed());
    }
    LinkInfoMessage::new(
        width,
        policy.next_index(),
        Some(StoredAddress::new(512)),
        Some(StoredAddress::new(1024)),
        if policy.is_indexed() {
            LinkCreationOrderIndex::Defined(StoredAddress::new(2048))
        } else {
            LinkCreationOrderIndex::Absent
        },
    )
    .unwrap()
}
