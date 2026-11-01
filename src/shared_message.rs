//! Resolves shared object header message references.
//!
//! Readers resolve references to object headers or objects in shared-message heaps.
//! Resolving a heap reference requires the file's shared-message table.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

pub use hdf5_pure_format::__private::DatatypeLocation;
pub use hdf5_pure_format::__private::FHEAP_ID_LEN;
pub use hdf5_pure_format::__private::SharedLocation;
pub use hdf5_pure_format::__private::SharedResolver;
pub use hdf5_pure_format::__private::committed_address_in_with_widths;
#[cfg(any(feature = "std", test))]
pub use hdf5_pure_format::__private::encode_committed_ref;
#[cfg(any(feature = "std", test))]
pub use hdf5_pure_format::__private::encode_sohm_ref;
pub use hdf5_pure_format::__private::parse_shared_ref_with_widths;

use crate::access_mode::AccessMode;
#[cfg(test)]
use crate::address::AddressContext;
#[cfg(test)]
use crate::address::BaseAddress;
use crate::address::StoredAddress;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::sohm::SohmTable;
use crate::source::OriginalSource;
use crate::source::PhysicalSource;
use crate::source::SourceMetadata;
use crate::source::StoredBytes;
use crate::width::FormatWidths;

/// Resolves shared messages by borrowing the resident image at stored addresses.
pub(crate) struct StoredResolver<'a> {
    source: StoredBytes<'a>,
    access_mode: AccessMode,
    widths: FormatWidths,
    sohm: Option<&'a SohmTable>,
}

impl<'a> StoredResolver<'a> {
    /// Creates a resolver that borrows resident object headers and heap metadata.
    ///
    /// `sohm` is the file's shared-message table. A heap reference requires this table.
    /// Read-write access requires complete validation of checksummed direct blocks for managed
    /// heap objects.
    pub fn new(
        source: StoredBytes<'a>,
        access_mode: AccessMode,
        widths: FormatWidths,
        sohm: Option<&'a SohmTable>,
    ) -> Self {
        Self {
            source,
            access_mode,
            widths,
            sohm,
        }
    }
}

impl SharedResolver for StoredResolver<'_> {
    fn resolve(&self, reference: &[u8], target: MessageType) -> Result<Vec<u8>, FormatError> {
        let Self {
            source,
            access_mode,
            widths,
            sohm,
        } = *self;
        let parsed = parse_shared_ref_with_widths(reference, widths)?;
        let addr = match parsed.location {
            SharedLocation::SohmHeap(id) => {
                let table = sohm.ok_or(FormatError::UnsupportedSohmReference)?;
                return crate::sohm::read_heap_message(
                    source,
                    access_mode,
                    table,
                    target,
                    &id,
                    widths,
                );
            }
            SharedLocation::ObjectHeader(addr) => addr,
        };
        let header = ObjectHeader::parse_with_widths(
            source.metadata_frame().bytes(),
            access_mode,
            addr.get().to_usize()?,
            widths,
        )?;
        select_shared_message(&header, target, addr.get())
    }

    fn committed_address(&self, reference: &[u8]) -> Result<Option<StoredAddress>, FormatError> {
        committed_address_in_with_widths(reference, self.widths)
    }
}

/// Resolves shared messages using the opened source's effective address context.
///
/// Reads object headers and heap objects from the original backend. Heap
/// references require a [`SohmTable`]. Read-write access requires complete validation of
/// checksummed direct blocks for managed heap objects.
pub(crate) struct SourceResolver<'a, S: OriginalSource + ?Sized> {
    source: &'a S,
    resident: Option<StoredBytes<'a>>,
    access_mode: AccessMode,
    widths: FormatWidths,
    sohm: Option<&'a SohmTable>,
}

impl<'a, S: OriginalSource + ?Sized> SourceResolver<'a, S> {
    /// Creates a resolver that reads through the original source's metadata interface.
    pub(crate) fn new(
        source: &'a S,
        access_mode: AccessMode,
        widths: FormatWidths,
        sohm: Option<&'a SohmTable>,
    ) -> Self {
        Self {
            source,
            resident: None,
            access_mode,
            widths,
            sohm,
        }
    }

    /// Creates a resolver that borrows resident bytes when the backend provides them.
    ///
    /// Reads through the original source's metadata interface when resident bytes are unavailable.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedSourceEof`] if the effective base exceeds the resident
    /// image's byte length.
    pub(crate) fn with_resident(
        source: &'a S,
        access_mode: AccessMode,
        widths: FormatWidths,
        sohm: Option<&'a SohmTable>,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            resident: PhysicalSource::new(source)
                .resident()
                .map(|bytes| bytes.stored())
                .transpose()?,
            ..Self::new(source, access_mode, widths, sohm)
        })
    }
}

impl<S: OriginalSource + ?Sized> SharedResolver for SourceResolver<'_, S> {
    fn resolve(&self, reference: &[u8], target: MessageType) -> Result<Vec<u8>, FormatError> {
        let Self {
            source,
            resident,
            access_mode,
            widths,
            sohm,
        } = *self;
        if let Some(resident) = resident {
            return StoredResolver::new(resident, access_mode, widths, sohm)
                .resolve(reference, target);
        }
        let physical = PhysicalSource::new(source);
        let parsed = parse_shared_ref_with_widths(reference, widths)?;
        let address = match parsed.location {
            SharedLocation::SohmHeap(id) => {
                let table = sohm.ok_or(FormatError::UnsupportedSohmReference)?;
                return crate::sohm::read_heap_message_from_source(
                    &physical.stored(),
                    access_mode,
                    table,
                    target,
                    &id,
                    widths,
                );
            }
            SharedLocation::ObjectHeader(address) => address,
        };
        let addresses = source.addresses();
        let header = ObjectHeader::parse_from_source_with_widths(
            &SourceMetadata(source),
            access_mode,
            addresses.resolve(address)?.get(),
            widths,
            addresses.effective_base(),
        )?;
        select_shared_message(&header, target, address.get())
    }

    fn committed_address(&self, reference: &[u8]) -> Result<Option<StoredAddress>, FormatError> {
        committed_address_in_with_widths(reference, self.widths)
    }
}

/// Returns the first inline message of `target_msg_type` in the target header.
///
/// # Errors
///
/// Returns [`FormatError::SharedMessageMissing`] if the target header has no inline message of
/// that type.
fn select_shared_message(
    target_header: &ObjectHeader,
    target_msg_type: MessageType,
    object_header_address: u64,
) -> Result<Vec<u8>, FormatError> {
    target_header
        .messages
        .iter()
        .find(|msg| msg.msg_type == target_msg_type && !msg.flags.is_shared())
        .map(|msg| msg.data.clone())
        .ok_or(FormatError::SharedMessageMissing {
            object_header_address,
            message_type: target_msg_type.to_u16(),
        })
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "std")]
    use rstest::rstest;
    #[cfg(feature = "std")]
    use test_util::object_header::Message;
    #[cfg(feature = "std")]
    use test_util::object_header::MessageType as RecordType;
    #[cfg(feature = "std")]
    use test_util::object_header::v2;

    use super::*;
    #[cfg(feature = "std")]
    use crate::File;
    #[cfg(feature = "std")]
    use crate::FileBuilder;
    #[cfg(feature = "std")]
    use crate::UserblockSize;
    #[cfg(feature = "std")]
    use crate::address::BaseAddressExt;
    #[cfg(feature = "std")]
    use crate::address::StoredAddressExt;
    use crate::message_flags::MessageFlags;
    use crate::object_header::HeaderMessage;
    #[cfg(feature = "std")]
    use crate::reader::OpenedSource;
    use crate::source;
    #[cfg(feature = "std")]
    use crate::source::BytesSource;

    #[cfg(feature = "std")]
    #[derive(Clone, Copy, Debug)]
    enum InvalidReference {
        AddressOverflow,
        MissingMessage,
        MissingTable,
        TruncatedHeader,
        TruncatedReference,
    }

    #[cfg(feature = "std")]
    #[rstest]
    #[case::address_overflow(InvalidReference::AddressOverflow)]
    #[case::missing_message(InvalidReference::MissingMessage)]
    #[case::missing_table(InvalidReference::MissingTable)]
    #[case::truncated_header(InvalidReference::TruncatedHeader)]
    #[case::truncated_reference(InvalidReference::TruncatedReference)]
    fn opened_resolvers_preserve_exact_reference_errors(
        #[case] invalid: InvalidReference,
        #[values(0, 512)] base: usize,
    ) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let mut header = v2::Header::new()
            .message(Message::new(RecordType::DATASPACE, &[1, 2, 3]))
            .build();
        let mut reference = encode_committed_ref(StoredAddress::new(64), widths.offsets);
        let expected = match invalid {
            InvalidReference::AddressOverflow => {
                reference = encode_committed_ref(StoredAddress::new(u64::MAX), widths.offsets);
                FormatError::OffsetOverflow {
                    offset: u64::MAX,
                    length: if base == 0 { 4 } else { base as u64 },
                }
            }
            InvalidReference::MissingMessage => FormatError::SharedMessageMissing {
                object_header_address: 64,
                message_type: MessageType::DATATYPE.to_u16(),
            },
            InvalidReference::MissingTable => {
                reference = encode_sohm_ref(&[0xFF; 8]);
                FormatError::UnsupportedSohmReference
            }
            InvalidReference::TruncatedHeader => {
                let expected = (base + 64 + header.len()) as u64;
                header.pop();
                FormatError::UnexpectedSourceEof {
                    expected,
                    available: expected - 1,
                }
            }
            InvalidReference::TruncatedReference => {
                reference.clear();
                FormatError::UnexpectedEof {
                    expected: 2,
                    available: 0,
                }
            }
        };
        let mut image = vec![0; base + 64];
        image.extend_from_slice(&header);
        let source = BytesSource::new(image);
        let opened = OpenedSource::for_test(
            &source,
            AddressContext::from_effective_base(BaseAddress::new(base as u64)),
        );
        let resolver = SourceResolver::new(&opened, AccessMode::ReadOnly, widths, None);
        assert_eq!(
            resolver
                .resolve(&reference, MessageType::DATATYPE)
                .unwrap_err(),
            expected
        );
    }

    #[cfg(feature = "std")]
    #[rstest]
    #[case::past_end(false)]
    #[case::maximum_address(true)]
    fn resident_resolvers_preserve_byte_parser_bounds(
        #[case] maximum: bool,
        #[values(0, 512)] base: usize,
    ) {
        let mut builder = FileBuilder::new();
        builder.with_userblock(UserblockSize::try_from(base as u64).unwrap());
        let image = builder.finish().unwrap();
        let available = image.len() - base;
        let address = if maximum {
            u64::MAX
        } else {
            available as u64 + 1
        };
        let expected = if let Ok(offset) = usize::try_from(address) {
            FormatError::UnexpectedEof {
                expected: offset.saturating_add(4),
                available,
            }
        } else {
            FormatError::ValueTooLargeForPlatform {
                value: address,
                target: "usize",
            }
        };
        let file = File::from_bytes(image).unwrap();
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let reference = encode_committed_ref(StoredAddress::new(address), widths.offsets);
        let error = file.with_opened_source(|source| {
            SourceResolver::with_resident(source, AccessMode::ReadOnly, widths, None)
                .unwrap()
                .resolve(&reference, MessageType::DATATYPE)
                .unwrap_err()
        });
        assert_eq!(error, expected);
        file.close().unwrap();
    }

    fn header_with(messages: Vec<HeaderMessage>) -> ObjectHeader {
        ObjectHeader {
            version: 2,
            messages,
            reference_count: None,
            flags: 0,
            access_time: None,
            modification_time: None,
            change_time: None,
            birth_time: None,
        }
    }

    fn message(msg_type: MessageType, flags: MessageFlags, data: Vec<u8>) -> HeaderMessage {
        HeaderMessage {
            msg_type,
            size: data.len(),
            flags,
            creation_order: None,
            data,
        }
    }

    #[test]
    fn a_heap_reference_without_a_table_is_rejected() {
        let reference = encode_sohm_ref(&[0xFF; 8]);
        let resolver = StoredResolver::new(
            source::zero_base_bytes(&[]),
            AccessMode::ReadOnly,
            FormatWidths {
                offsets: crate::width::OffsetWidth::Eight,
                lengths: crate::width::LengthWidth::Eight,
            },
            None,
        );

        let err = resolver
            .resolve(&reference, MessageType::DATATYPE)
            .unwrap_err();
        assert_eq!(err, FormatError::UnsupportedSohmReference);
    }

    #[test]
    fn a_reference_to_a_header_without_that_message_is_an_error() {
        let header = header_with(vec![message(
            MessageType::DATASPACE,
            MessageFlags::NONE,
            vec![1, 2, 3],
        )]);
        let err = select_shared_message(&header, MessageType::DATATYPE, 0x320).unwrap_err();
        assert_eq!(
            err,
            FormatError::SharedMessageMissing {
                object_header_address: 0x320,
                message_type: MessageType::DATATYPE.to_u16(),
            }
        );
    }

    #[test]
    fn a_shared_message_in_the_target_is_not_mistaken_for_content() {
        let header = header_with(vec![message(
            MessageType::DATATYPE,
            MessageFlags::SHARED,
            vec![2, 2, 0, 0],
        )]);
        let err = select_shared_message(&header, MessageType::DATATYPE, 0x320).unwrap_err();
        assert_eq!(
            err,
            FormatError::SharedMessageMissing {
                object_header_address: 0x320,
                message_type: MessageType::DATATYPE.to_u16(),
            }
        );
    }
}
