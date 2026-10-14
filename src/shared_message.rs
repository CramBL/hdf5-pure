//! HDF5 Shared Object Header Message resolution.
//!
//! A header message whose record has the shared flag (bit 1 of `msg_flags`) set
//! does not hold its own content. Its body is a *reference* to the one copy of
//! that message stored elsewhere, and the same reference encoding appears inside
//! an attribute message whose datatype or dataspace field is shared.
//!
//! Two things can be on the other end of a reference:
//!
//! - another **object header**, which is what `H5Tcommit` writes for a named
//!   ("committed") datatype — resolved here;
//! - the file's **shared object header message (SOHM) heap**, a fractal heap the
//!   file's shared-message table names — resolved through [`crate::sohm`] when the
//!   resolver was given that table, and refused by name rather than mis-read when
//!   it was not.

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
use crate::address::AddressContext;
use crate::address::BaseAddressExt;
use crate::address::{BaseAddress, StoredAddress};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::sohm::SohmTable;
use crate::source::Source;
use crate::source::SourceMetadata;
use crate::width::FormatWidths;

/// Resolves references against a whole-file slice, already framed at the file's
/// base address (shared-message addresses are stored relative to it).
pub struct BufferedResolver<'a> {
    file_data: &'a [u8],
    addresses: AddressContext,
    access_mode: AccessMode,
    widths: FormatWidths,
    sohm: Option<&'a SohmTable>,
}

impl<'a> BufferedResolver<'a> {
    /// Creates a resolver for an image framed at the effective base in `addresses`.
    ///
    /// `sohm` is the file's shared-message table. A heap reference requires this table.
    /// Read-write access requires complete direct-block checksum validation for heap messages.
    pub fn new(
        file_data: &'a [u8],
        addresses: AddressContext,
        access_mode: AccessMode,
        widths: FormatWidths,
        sohm: Option<&'a SohmTable>,
    ) -> Self {
        Self {
            file_data,
            addresses,
            access_mode,
            widths,
            sohm,
        }
    }
}

impl SharedResolver for BufferedResolver<'_> {
    fn resolve(&self, reference: &[u8], target: MessageType) -> Result<Vec<u8>, FormatError> {
        let parsed = parse_shared_ref_with_widths(reference, self.widths)?;
        let addr = match parsed.location {
            SharedLocation::SohmHeap(id) => {
                let table = self.sohm.ok_or(FormatError::UnsupportedSohmReference)?;
                return crate::sohm::read_heap_message(
                    self.file_data,
                    self.access_mode,
                    self.addresses,
                    table,
                    target,
                    &id,
                    self.widths,
                );
            }
            SharedLocation::ObjectHeader(addr) => addr,
        };
        let header = ObjectHeader::parse_with_widths(
            self.file_data,
            self.access_mode,
            addr.get().to_usize()?,
            self.widths,
        )?;
        select_shared_message(&header, target, addr.get())
    }

    fn committed_address(&self, reference: &[u8]) -> Result<Option<StoredAddress>, FormatError> {
        committed_address_in_with_widths(reference, self.widths)
    }
}

/// Resolves shared messages by reading object headers and heap objects through a [`Source`].
///
/// The source is framed at the effective base in [`AddressContext`].
pub struct SourceResolver<'a, S: Source + ?Sized> {
    source: &'a S,
    addresses: AddressContext,
    access_mode: AccessMode,
    widths: FormatWidths,
    sohm: Option<&'a SohmTable>,
}

impl<'a, S: Source + ?Sized> SourceResolver<'a, S> {
    /// Creates a resolver for a source framed at the effective base in `addresses`.
    ///
    /// `sohm` is the file's shared-message table. A heap reference requires this table.
    /// Read-write access requires complete direct-block checksum validation for heap messages.
    pub fn new(
        source: &'a S,
        addresses: AddressContext,
        access_mode: AccessMode,
        widths: FormatWidths,
        sohm: Option<&'a SohmTable>,
    ) -> Self {
        Self {
            source,
            addresses,
            access_mode,
            widths,
            sohm,
        }
    }
}

impl<S: Source + ?Sized> SharedResolver for SourceResolver<'_, S> {
    fn resolve(&self, reference: &[u8], target: MessageType) -> Result<Vec<u8>, FormatError> {
        let parsed = parse_shared_ref_with_widths(reference, self.widths)?;
        let addr = match parsed.location {
            SharedLocation::SohmHeap(id) => {
                let table = self.sohm.ok_or(FormatError::UnsupportedSohmReference)?;
                return crate::sohm::read_heap_message_from_source(
                    self.source,
                    self.access_mode,
                    self.addresses,
                    table,
                    target,
                    &id,
                    self.widths,
                );
            }
            SharedLocation::ObjectHeader(addr) => addr,
        };
        // base_address 0 matches the buffered path, whose slice is already framed
        // at the base address, so both treat the reference as absolute within it.
        let header = ObjectHeader::parse_from_source_with_widths(
            &SourceMetadata(self.source),
            self.access_mode,
            addr.get(),
            self.widths,
            BaseAddress::ZERO,
        )?;
        select_shared_message(&header, target, addr.get())
    }

    fn committed_address(&self, reference: &[u8]) -> Result<Option<StoredAddress>, FormatError> {
        committed_address_in_with_widths(reference, self.widths)
    }
}

/// Pick the message of `target_msg_type` out of a resolved target object header.
///
/// Only a message that carries its own content will do: one that is itself a
/// reference would hand back reference bytes for the caller to decode as content,
/// which is the defect this module exists to prevent.
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
    use super::*;
    use crate::message_flags::MessageFlags;
    use crate::object_header::HeaderMessage;

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

    /// A resolver given no shared-message table refuses a heap reference by
    /// name. Its heap id is not an address, so the alternative is an
    /// object-header parse at whatever those eight bytes spell.
    #[test]
    fn a_sohm_reference_without_a_table_is_refused_rather_than_followed() {
        let reference = encode_sohm_ref(&[0xFF; 8]);
        let resolver = BufferedResolver::new(
            &[],
            AddressContext::from_effective_base(BaseAddress::new(0)),
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

    /// The target header must hold the message the reference stands in for.
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

    /// A message that is itself a reference is not content, so it is not an
    /// answer: handing its bytes back would re-create the mis-decode one level
    /// down.
    #[test]
    fn a_shared_message_in_the_target_is_not_mistaken_for_content() {
        let header = header_with(vec![message(
            MessageType::DATATYPE,
            MessageFlags::SHARED,
            vec![2, 2, 0, 0],
        )]);
        let err = select_shared_message(&header, MessageType::DATATYPE, 0x320).unwrap_err();
        assert!(matches!(err, FormatError::SharedMessageMissing { .. }));
    }
}
