use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::file_space_info;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType;
use test_util::object_header::v1;
use test_util::object_header::v2;
use test_util::sohm;
use test_util::widths::Widths;

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::error::FormatError;
use crate::extension::ExtensionSummary;
use crate::extension::SharedTableSummary;
use crate::file_space_info::FileSpaceStrategy;
use crate::source::BytesSource;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

#[derive(Clone, Copy)]
enum BodyError {
    FileSpace,
    SharedTable,
}

#[derive(Clone, Copy)]
enum HeaderVersion {
    Legacy,
    Modern,
}

#[rstest]
#[case::v1(None, HeaderVersion::Legacy)]
#[case::v2(None, HeaderVersion::Modern)]
#[case::file_space(Some(BodyError::FileSpace), HeaderVersion::Modern)]
#[case::shared_table(Some(BodyError::SharedTable), HeaderVersion::Modern)]
fn extension_facts_preserve_independent_message_results(
    #[case] error: Option<BodyError>,
    #[case] version: HeaderVersion,
) {
    let mut file_space = file_space_info::version_0(
        Widths::EIGHT,
        file_space_info::ALL_PERSIST,
        7,
        &[u64::MAX; file_space_info::VERSION_0_MANAGERS],
    );
    let mut table = sohm::table_message(8_192, 2, Widths::EIGHT);
    match error {
        Some(BodyError::FileSpace) => file_space[0] = 99,
        Some(BodyError::SharedTable) => table[0] = 99,
        None => {}
    }
    let messages = [
        Message::new(MessageType::FILE_SPACE_INFO, &file_space),
        Message::new(MessageType::SHARED_MESSAGE_TABLE, &table),
        Message::new(MessageType::FILE_SPACE_INFO, &[98]),
        Message::new(MessageType::SHARED_MESSAGE_TABLE, &[98]),
    ];
    let bytes = match version {
        HeaderVersion::Modern => v2::Header::new().messages(messages).build(),
        HeaderVersion::Legacy => v1::Header::new()
            .messages(messages.map(|mut message| {
                message.data.resize(
                    message.data.len().div_ceil(v1::MESSAGE_ALIGNMENT) * v1::MESSAGE_ALIGNMENT,
                    0,
                );
                message
            }))
            .build(),
    };
    let source = BytesSource::new(bytes);
    let facts = ExtensionSummary::read(
        &source,
        AccessMode::ReadWrite,
        PhysicalAddress::new(0),
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        },
        AddressContext::from_effective_base(BaseAddress::ZERO),
        None,
    )
    .unwrap();
    match error {
        Some(BodyError::FileSpace) => assert!(
            matches!(
                facts.file_space,
                Err(FormatError::UnsupportedFileSpaceInfoVersion(99))
            ),
            "{:?}",
            facts.file_space
        ),
        _ => {
            let info = facts.file_space.unwrap().unwrap();
            assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
            assert!(info.persist);
            assert_eq!(info.threshold, 7);
            assert_eq!(
                info.manager_addrs.len(),
                file_space_info::VERSION_0_MANAGERS
            );
            assert_eq!(
                info.manager_addrs,
                vec![DecodedStoredAddress::Undefined; file_space_info::VERSION_0_MANAGERS]
            );
            assert_eq!(info.page_size.get(), 4_096);
            assert_eq!(info.page_end_meta_threshold, 0);
            assert_eq!(info.eoa_pre_fsm, None);
        }
    }
    match error {
        Some(BodyError::SharedTable) => assert!(
            matches!(
                facts.shared_table,
                Err(FormatError::InvalidSohmTableVersion(99))
            ),
            "{:?}",
            facts.shared_table
        ),
        _ => {
            let table = facts.shared_table.unwrap().unwrap();
            assert_eq!(table.table_address, StoredAddress::new(8_192));
            assert_eq!(table.index_count, 2);
        }
    }
}

#[rstest]
#[case::combined(false)]
#[case::shared_table_only(true)]
fn a_later_header_error_precedes_a_pending_message_error(#[case] shared_only: bool) {
    let source = BytesSource::new(
        v2::Header::new()
            .message(Message::new(MessageType::FILE_SPACE_INFO, &[99]))
            .message(
                Message::new(MessageType::UNKNOWN, &[0xA5])
                    .with_flags(MessageFlags::FAIL_IF_UNKNOWN_ALWAYS),
            )
            .build(),
    );
    let address = PhysicalAddress::new(0);
    let widths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };
    let addresses = AddressContext::from_effective_base(BaseAddress::ZERO);
    let error = if shared_only {
        SharedTableSummary::read(&source, AccessMode::ReadWrite, address, widths, addresses)
            .err()
            .unwrap()
    } else {
        ExtensionSummary::read(
            &source,
            AccessMode::ReadWrite,
            address,
            widths,
            addresses,
            None,
        )
        .err()
        .unwrap()
    };
    let FormatError::UnsupportedMessage(kind) = error else {
        panic!("expected UnsupportedMessage, got {error:?}");
    };
    assert_eq!(kind, MessageType::UNKNOWN.0);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn a_shared_table_query_skips_file_space_decoding_and_allocation() {
    let source = BytesSource::new(
        v2::Header::new()
            .message(Message::new(
                MessageType::FILE_SPACE_INFO,
                &file_space_info::version_0(
                    Widths::EIGHT,
                    file_space_info::ALL_PERSIST,
                    7,
                    &[u64::MAX; file_space_info::VERSION_0_MANAGERS],
                ),
            ))
            .message(Message::new(
                MessageType::SHARED_MESSAGE_TABLE,
                &sohm::table_message(8_192, 2, Widths::EIGHT),
            ))
            .message(Message::new(MessageType::SHARED_MESSAGE_TABLE, &[99]))
            .build(),
    );
    let address = PhysicalAddress::new(0);
    let widths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };
    let addresses = AddressContext::from_effective_base(BaseAddress::ZERO);
    let (selected, measured) = allocation::measure("shared_table_only", || {
        SharedTableSummary::read(&source, AccessMode::ReadWrite, address, widths, addresses)
            .unwrap()
    });
    let (combined, control) = allocation::measure("shared_table_combined_control", || {
        ExtensionSummary::read(
            &source,
            AccessMode::ReadWrite,
            address,
            widths,
            addresses,
            None,
        )
        .unwrap()
    });
    let selected = selected.into_message().unwrap().unwrap();
    assert_eq!(selected.table_address, StoredAddress::new(8_192));
    assert_eq!(selected.index_count, 2);
    assert_eq!(combined.shared_table.unwrap(), Some(selected));
    let info = combined.file_space.unwrap().unwrap();
    assert!(info.persist);
    assert_eq!(info.threshold, 7);
    assert_eq!(
        info.manager_addrs,
        vec![DecodedStoredAddress::Undefined; file_space_info::VERSION_0_MANAGERS]
    );
    assert!(control.blocks > 0, "{control}");
    assert!(
        control.live_bytes >= u64::try_from(file_space_info::VERSION_0_MANAGERS).unwrap(),
        "{control}"
    );
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.bytes, 0, "{measured}");
    assert_eq!(measured.live_bytes, 0, "{measured}");
}

#[test]
fn a_shared_table_query_preserves_readable_message_errors() {
    let source = BytesSource::new(
        v2::Header::new()
            .message(Message::new(MessageType::FILE_SPACE_INFO, &[98]))
            .message(Message::new(MessageType::SHARED_MESSAGE_TABLE, &[99]))
            .build(),
    );
    let summary = SharedTableSummary::read(
        &source,
        AccessMode::ReadWrite,
        PhysicalAddress::new(0),
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        },
        AddressContext::from_effective_base(BaseAddress::ZERO),
    )
    .unwrap();
    let error = summary.into_message().unwrap_err();
    assert!(
        matches!(error, FormatError::InvalidSohmTableVersion(99)),
        "{error:?}"
    );
}
