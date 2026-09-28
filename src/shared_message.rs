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

pub use hdf5_pure_format::FHEAP_ID_LEN;
pub use hdf5_pure_format::SharedLocation;
pub use hdf5_pure_format::SharedResolver;
pub use hdf5_pure_format::Unresolvable;
pub use hdf5_pure_format::committed_address_in;
pub use hdf5_pure_format::encode_committed_ref;
#[cfg(any(feature = "std", test))]
pub use hdf5_pure_format::encode_sohm_ref;
pub use hdf5_pure_format::parse_shared_ref;

use crate::access_mode::AccessMode;
use crate::address::BaseAddressExt;
use crate::address::{BaseAddress, StoredAddress};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::sohm::SohmTable;
use crate::source::Source;
use crate::source::SourceMetadata;
use crate::width::OffsetWidth;

/// Where a datatype is stored, for a message that could hold it either way.
///
/// A datatype is the one part of a dataset or attribute that can live outside
/// the message describing it: `H5Tcommit` puts it in its own object header, and
/// everything using it carries a reference in place of the encoding. Both forms
/// decode to the same [`Datatype`](crate::Datatype), so this is what
/// separates a message that *refers to* a type from one that spells it out, a
/// distinction `h5dump` reports, and a rewrite has to preserve.
///
/// No `Default`: an omitted location silently means `Inline`, and a reference
/// that decodes as an encoding is the whole defect this type exists to prevent.
/// Every construction states its variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatatypeLocation {
    /// Encoded in the message itself.
    Inline,
    /// A reference to the committed datatype object at this address, in the file
    /// the message belongs to. What a parse reads out of a file, and what a
    /// writer emits once the object's address is fixed.
    Committed(StoredAddress),
}

impl DatatypeLocation {
    /// The reference body to write in place of the datatype encoding, or `None`
    /// when the datatype is written inline.
    pub fn reference_bytes(&self, offset_width: OffsetWidth) -> Option<Vec<u8>> {
        match self {
            Self::Inline => None,
            Self::Committed(addr) => Some(encode_committed_ref(*addr, offset_width)),
        }
    }

    /// Whether the datatype lives in a committed object and not in the
    /// message.
    pub fn is_committed(&self) -> bool {
        !matches!(self, Self::Inline)
    }
}

/// Resolves references against a whole-file slice, already framed at the file's
/// base address (shared-message addresses are stored relative to it).
pub struct BufferedResolver<'a> {
    file_data: &'a [u8],
    access_mode: AccessMode,
    offset_size: u8,
    length_size: u8,
    sohm: Option<&'a SohmTable>,
}

impl<'a> BufferedResolver<'a> {
    /// `sohm` is the file's shared-message table, which only a file created with
    /// `H5Pset_shared_mesg_index` has. It is a parameter rather than a default
    /// because a resolver without it refuses every heap-stored message, and a
    /// caller that has the table and forgets to pass it would turn a readable
    /// file into an unreadable one silently.
    pub fn new(
        file_data: &'a [u8],
        access_mode: AccessMode,
        offset_size: u8,
        length_size: u8,
        sohm: Option<&'a SohmTable>,
    ) -> Self {
        Self {
            file_data,
            access_mode,
            offset_size,
            length_size,
            sohm,
        }
    }
}

impl SharedResolver for BufferedResolver<'_> {
    fn resolve(&self, reference: &[u8], target: MessageType) -> Result<Vec<u8>, FormatError> {
        let parsed = parse_shared_ref(reference, self.offset_size, self.length_size)?;
        let addr = match parsed.location {
            SharedLocation::SohmHeap(id) => {
                let table = self.sohm.ok_or(FormatError::UnsupportedSohmReference)?;
                return crate::sohm::read_heap_message(
                    self.file_data,
                    table,
                    target,
                    &id,
                    self.offset_size,
                    self.length_size,
                );
            }
            SharedLocation::ObjectHeader(addr) => addr,
        };
        let header = ObjectHeader::parse(
            self.file_data,
            self.access_mode,
            addr.get().to_usize()?,
            self.offset_size,
            self.length_size,
        )?;
        select_shared_message(&header, target, addr.get())
    }

    fn committed_address(&self, reference: &[u8]) -> Result<Option<StoredAddress>, FormatError> {
        committed_address_in(reference, self.offset_size, self.length_size)
    }
}

/// Resolves references by reading the target object header from a [`Source`] on
/// demand instead of indexing a whole-file slice.
pub struct SourceResolver<'a, S: Source + ?Sized> {
    source: &'a S,
    access_mode: AccessMode,
    offset_size: u8,
    length_size: u8,
    sohm: Option<&'a SohmTable>,
}

impl<'a, S: Source + ?Sized> SourceResolver<'a, S> {
    /// See [`BufferedResolver::new`] for why the shared-message table is a
    /// parameter here rather than something the resolver finds for itself.
    pub fn new(
        source: &'a S,
        access_mode: AccessMode,
        offset_size: u8,
        length_size: u8,
        sohm: Option<&'a SohmTable>,
    ) -> Self {
        Self {
            source,
            access_mode,
            offset_size,
            length_size,
            sohm,
        }
    }
}

impl<S: Source + ?Sized> SharedResolver for SourceResolver<'_, S> {
    fn resolve(&self, reference: &[u8], target: MessageType) -> Result<Vec<u8>, FormatError> {
        let parsed = parse_shared_ref(reference, self.offset_size, self.length_size)?;
        let addr = match parsed.location {
            SharedLocation::SohmHeap(id) => {
                let table = self.sohm.ok_or(FormatError::UnsupportedSohmReference)?;
                return crate::sohm::read_heap_message_from_source(
                    self.source,
                    table,
                    target,
                    &id,
                    self.offset_size,
                    self.length_size,
                );
            }
            SharedLocation::ObjectHeader(addr) => addr,
        };
        // base_address 0 matches the buffered path, whose slice is already framed
        // at the base address, so both treat the reference as absolute within it.
        let header = ObjectHeader::parse_from_source(
            &SourceMetadata(self.source),
            self.access_mode,
            addr.get(),
            self.offset_size,
            self.length_size,
            BaseAddress::ZERO,
        )?;
        select_shared_message(&header, target, addr.get())
    }

    fn committed_address(&self, reference: &[u8]) -> Result<Option<StoredAddress>, FormatError> {
        committed_address_in(reference, self.offset_size, self.length_size)
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
        let resolver = BufferedResolver::new(&[], AccessMode::ReadOnly, 8, 8, None);

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
