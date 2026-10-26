#![cfg(feature = "hdf5")]

use std::collections::BTreeMap;
use std::path::Path;

use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::{Sizeof, SizeofInfo};
use hdf5::plist::group_create::LinkCreationOrder;
use hdf5::plist::link_create::CharEncoding;
#[cfg(feature = "__hdf5-1.14")]
use hdf5::{MajorErrorCode, MinorErrorCode};
use hdf5_pure::{AttrValue, CharacterSet, Error, File, MessageType};
use hdf5_pure_format::__private::{
    AccessMode, GroupLinkStorage, LinkCreationCounter, LinkCreationPolicy, LinkInfoMessage,
    ObjectHeader, ObjectHeaderPrefix, OffsetWidth, OwnedLink, OwnedLinkTarget,
};
use rstest::rstest;

#[rstest]
#[case::untracked(LinkCreationOrder::Untracked)]
#[case::tracked(LinkCreationOrder::Tracked)]
#[case::indexed(LinkCreationOrder::Indexed)]
fn libhdf5_link_messages_preserve_targets_encodings_and_creation_policy(
    #[case] order: LinkCreationOrder,
    #[values(Sizeof::Bytes2, Sizeof::Bytes4, Sizeof::Bytes8)] address_size: Sizeof,
    #[values(false, true)] deleted: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("links.h5");
    let file = hdf5::File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V18, LibraryVersion::latest())
        })
        .with_fcpl(|properties| {
            properties.sizes(SizeofInfo {
                sizeof_addr: address_size,
                sizeof_size: address_size,
            });
            properties.link_creation_order(order)
        })
        .create(&path)
        .unwrap();
    file.create_group("hard").unwrap();
    file.link_soft("/hard", "soft").unwrap();
    file.link_external("外部.h5", "/目标", "external").unwrap();
    file.link_hard("/hard", "别名").unwrap();
    file.create_group_builder()
        .char_encoding(CharEncoding::Utf8)
        .create("utf8_链")
        .unwrap();
    if deleted {
        file.link_soft("/hard", "deleted").unwrap();
        file.unlink("deleted").unwrap();
    }
    file.close().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let file = File::open(&path).unwrap();
    let superblock = file.superblock();
    let width = superblock.offset_size;
    let header = ObjectHeader::parse(
        &bytes,
        AccessMode::ReadOnly,
        usize::try_from(superblock.root_group_address.get()).unwrap(),
        width.get(),
        superblock.length_size.get(),
    )
    .unwrap();
    drop(file);
    let message = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::LINK_INFO)
        .unwrap();
    let info = LinkInfoMessage::parse_with_offset_width(&message.data, width).unwrap();
    let links = header
        .messages
        .iter()
        .filter(|message| message.msg_type == MessageType::LINK)
        .map(|message| {
            let link = OwnedLink::parse(&message.data, width).unwrap();
            assert_eq!(link.serialize(width).unwrap(), message.data);
            link
        })
        .collect::<Vec<_>>();
    let mut storage = GroupLinkStorage::parse(&message.data, links, width).unwrap();
    assert_eq!(storage.info(), info);
    assert_eq!(info.fractal_heap_address, None);
    assert_eq!(info.btree_name_index_address, None);
    let tracked = order != LinkCreationOrder::Untracked;
    let expected_next = 5 + u64::from(deleted);
    let policy = match order {
        LinkCreationOrder::Untracked => LinkCreationPolicy::Untracked,
        LinkCreationOrder::Tracked => LinkCreationPolicy::Tracked {
            next: LinkCreationCounter::new(expected_next).unwrap(),
        },
        LinkCreationOrder::Indexed => LinkCreationPolicy::Indexed {
            next: LinkCreationCounter::new(expected_next).unwrap(),
        },
    };
    assert_eq!(storage.creation_order(), policy);
    assert_eq!(
        info.btree_creation_order_address,
        policy.is_indexed().then_some(None)
    );
    let links = storage
        .links()
        .iter()
        .map(|link| (link.name().to_vec(), link))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        links.keys().cloned().collect::<Vec<_>>(),
        [
            b"external".to_vec(),
            b"hard".to_vec(),
            b"soft".to_vec(),
            "utf8_链".as_bytes().to_vec(),
            "别名".as_bytes().to_vec()
        ]
    );
    assert_eq!(
        links[b"hard".as_slice()].target(),
        links["别名".as_bytes()].target()
    );
    assert!(matches!(
        links[b"hard".as_slice()].target(),
        OwnedLinkTarget::Hard { .. }
    ));
    assert_eq!(
        links[b"soft".as_slice()].target(),
        &OwnedLinkTarget::Soft {
            target_path: b"/hard".to_vec()
        }
    );
    assert_eq!(
        links[b"external".as_slice()].target(),
        &OwnedLinkTarget::External {
            filename: "外部.h5".as_bytes().to_vec(),
            object_path: "/目标".as_bytes().to_vec()
        }
    );
    assert_eq!(links["别名".as_bytes()].charset(), CharacterSet::Ascii);
    assert_eq!(links["utf8_链".as_bytes()].charset(), CharacterSet::Utf8);
    assert_eq!(
        links["utf8_链".as_bytes()].creation_order(),
        tracked.then_some(4)
    );
    assert_eq!(
        links[b"hard".as_slice()].creation_order(),
        tracked.then_some(0)
    );
    assert_eq!(
        links[b"soft".as_slice()].creation_order(),
        tracked.then_some(1)
    );
    assert_eq!(
        links[b"external".as_slice()].creation_order(),
        tracked.then_some(2)
    );
    assert_eq!(
        links["别名".as_bytes()].creation_order(),
        tracked.then_some(3)
    );
    let target = links[b"hard".as_slice()].target().clone();
    storage
        .insert(OwnedLink::new(b"added".to_vec(), CharacterSet::Ascii, target, None).unwrap())
        .unwrap();
    assert_eq!(
        storage.links().last().unwrap().creation_order(),
        tracked.then_some(expected_next)
    );
    assert_eq!(
        storage.info().max_creation_order,
        tracked.then_some(expected_next + 1)
    );
    assert_eq!(
        width,
        OffsetWidth::try_from(u8::try_from(usize::from(address_size)).unwrap()).unwrap()
    );
}

#[rstest]
#[case::tracked(LinkCreationOrder::Tracked)]
#[case::indexed(LinkCreationOrder::Indexed)]
fn libhdf5_counter_boundaries_are_rejected_during_edit_preflight(
    #[case] order: LinkCreationOrder,
    #[values(0, 1)] remaining: u64,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("counter.h5");
    write_counter_fixture(&path, order, MAX_COUNTER - remaining - 1);
    let reference = hdf5::File::open_rw(&path).unwrap();
    reference.create_group("last").unwrap();
    assert_eq!(
        reference.info().unwrap().max_corder,
        i64::try_from(MAX_COUNTER - remaining).unwrap()
    );
    reference.close().unwrap();
    let before = std::fs::read(&path).unwrap();
    let error = {
        let file = File::open_rw(&path).unwrap();
        for index in 0..=remaining {
            file.root()
                .create_group(&format!("overflow{index}"))
                .unwrap();
        }
        file.commit().unwrap_err()
    };
    let Error::EditUnsupported(reason) = error else {
        panic!("expected exhausted link creation order, got {error:?}");
    };
    assert_eq!(
        reason,
        "a group has assigned every link creation index its link-info message can record"
    );
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let reference = hdf5::File::open(&path).unwrap();
    assert_eq!(reference.member_names().unwrap(), ["existing", "last"]);
    assert_eq!(
        reference.info().unwrap().max_corder,
        i64::try_from(MAX_COUNTER - remaining).unwrap()
    );
}

#[rstest]
#[case::exhausted_but_valid(MAX_COUNTER, true)]
#[case::outside_domain(MAX_COUNTER + 1, false)]
fn attribute_edits_preserve_exhausted_counters_and_reject_unreproducible_counters(
    #[case] counter: u64,
    #[case] supported: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("counter.h5");
    write_counter_fixture(&path, LinkCreationOrder::Tracked, counter);
    let before = std::fs::read(&path).unwrap();
    let result = {
        let file = File::open_rw(&path).unwrap();
        file.root().set_attr("note", AttrValue::I64(7)).unwrap();
        file.commit()
    };
    if supported {
        result.unwrap();
        let reference = hdf5::File::open(&path).unwrap();
        assert_eq!(reference.info().unwrap().max_corder, i64::MAX);
        assert_eq!(
            reference
                .attr("note")
                .unwrap()
                .read_scalar::<i64>()
                .unwrap(),
            7
        );
    } else {
        let error = result.unwrap_err();
        let Error::EditUnsupported(reason) = error else {
            panic!("expected unsupported link creation policy, got {error:?}");
        };
        assert_eq!(
            reason,
            "a group's link creation-order policy cannot be reproduced"
        );
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }
}

#[cfg(feature = "__hdf5-1.14")]
#[rstest]
#[case::sign_bit(MAX_COUNTER + 1)]
#[case::unsigned_last(u64::MAX - 1)]
#[case::unsigned_max(u64::MAX)]
fn libhdf5_rejects_counters_outside_the_nonnegative_signed_domain(#[case] counter: u64) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("counter.h5");
    write_counter_fixture(&path, LinkCreationOrder::Tracked, counter);
    let error = hdf5::File::open(&path)
        .and_then(|file| file.member_names())
        .unwrap_err();
    assert!(
        error.contains_major(MajorErrorCode::ObjectHeader),
        "{error:?}"
    );
    assert!(error.contains_minor(MinorErrorCode::BadValue), "{error:?}");
}

/// Writes a libhdf5 group and patches its Link Info counter with a recomputed header checksum.
fn write_counter_fixture(path: &Path, order: LinkCreationOrder, counter: u64) {
    let reference = hdf5::File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V18, LibraryVersion::latest())
        })
        .with_fcpl(|properties| properties.link_creation_order(order))
        .create(path)
        .unwrap();
    reference.create_group("existing").unwrap();
    reference.close().unwrap();
    let mut bytes = std::fs::read(path).unwrap();
    let file = File::open(path).unwrap();
    let root_at = usize::try_from(file.superblock().root_group_address.get()).unwrap();
    drop(file);
    let prefix = ObjectHeaderPrefix::parse(&bytes[root_at..]).unwrap();
    let end = root_at + prefix.len + usize::try_from(prefix.chunk0_size).unwrap();
    let layout = prefix.prefix.layout;
    let record = std::iter::successors(
        layout
            .next_message(&bytes[..end], root_at + prefix.len)
            .unwrap(),
        |record| {
            layout
                .next_message(&bytes[..end], record.body_range.end)
                .unwrap()
        },
    )
    .find(|record| record.msg_type == MessageType::LINK_INFO)
    .unwrap();
    let counter_at = record.body_range.start + LINK_INFO_COUNTER_OFFSET;
    test_util::bytes::set_uint_at(&mut bytes, counter_at, CREATION_COUNTER_LEN, counter);
    test_util::checksum::restamp(
        &mut bytes,
        root_at,
        end - root_at + test_util::checksum::CHECKSUM,
    );
    std::fs::write(path, bytes).unwrap();
}

/// The largest counter accepted by `H5O__linfo_decode` in `H5Olinfo.c` (HDF5 1.14.6 and 2.2.0).
const MAX_COUNTER: u64 = 0x7fff_ffff_ffff_ffff;
/// The counter offset after the version and flags of a Link Info message.
///
/// Defined in "The Link Info Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_linkinfo
const LINK_INFO_COUNTER_OFFSET: usize = 2;
/// The eight-byte Maximum Creation Index field from the same table as [`LINK_INFO_COUNTER_OFFSET`].
const CREATION_COUNTER_LEN: usize = 8;
