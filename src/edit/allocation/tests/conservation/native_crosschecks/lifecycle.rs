//! Crosschecks stored values, chunk records, and free ranges after injected append errors.

use std::fs;

use rstest::rstest;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Interval;
use test_util::bytes;
use test_util::data_layout;
use test_util::disk_log::FailurePoint;
use test_util_hdf5::allocation_publication;

use crate::File;
use crate::SyncPolicy;
use crate::access_mode::AccessMode;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;

use super::super::Observation;
use super::super::lifecycle;
use super::super::lifecycle::Action;
use super::super::lifecycle::Fault;
use super::super::lifecycle::Flush;
use super::super::lifecycle::WriteFault;

#[rstest]
#[case::immediate(Action::Immediate)]
#[case::borrowed_automatic(Action::Borrowed(Flush::Automatic))]
#[case::borrowed_manual(Action::Borrowed(Flush::Manual))]
#[case::borrowed_finish(Action::Borrowed(Flush::Finish))]
#[case::borrowed_drop(Action::Borrowed(Flush::Drop))]
#[case::owned_automatic(Action::Owned(Flush::Automatic))]
#[case::owned_manual(Action::Owned(Flush::Manual))]
#[case::owned_finish(Action::Owned(Flush::Finish))]
#[case::owned_drop(Action::Owned(Flush::Drop))]
fn native_indexes_preserve_lifecycle_failure_ownership(
    #[case] action: Action,
    #[values(false, true)] deferred: bool,
    #[values(false, true)] filtered: bool,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] policy: SyncPolicy,
    #[values(
        Fault::Content(WriteFault::Partial),
        Fault::Ownership(FailurePoint::After)
    )]
    fault: Fault,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-lifecycle.h5");
    let (address, length) = allocation_publication::write_fixture(&path, false, deferred, filtered);
    let hole = Coverage::new([Interval::new(address, length).unwrap()]);
    let initial = Observation::read(&fs::read(&path).unwrap());
    assert_eq!(initial.index_defined, !deferred);
    if deferred {
        let image = fs::read(&path).unwrap();
        let reader = File::from_bytes(image.clone()).unwrap();
        let header = reader.dataset("d").unwrap().header_address().unwrap();
        let object = ObjectHeader::parse(
            &image,
            AccessMode::ReadOnly,
            usize::try_from(header).unwrap(),
            8,
            8,
        )
        .unwrap();
        let layout = &object
            .messages
            .iter()
            .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
            .unwrap()
            .data;
        assert_eq!(layout[0], data_layout::VERSION_FOUR);
        assert_eq!(layout[1], data_layout::CHUNKED_CLASS);
        assert_eq!(
            bytes::u64_at(layout, layout.len() - size_of::<u64>()),
            data_layout::UNDEFINED_ADDRESS
        );
    }
    assert!(hole.difference(&initial.free).is_empty());
    lifecycle::failure_on(&path, action, policy, fault, &hole);
    let pure = Observation::read(&fs::read(&path).unwrap());
    let native = allocation_publication::Observation::read(&path);
    assert_eq!(pure.values, native.values);
    assert_eq!(pure.chunks, native.chunks);
    assert_eq!(
        pure.free,
        Coverage::new(
            native
                .free
                .into_iter()
                .map(|(address, length)| Interval::new(address, length).unwrap())
        )
    );
}
