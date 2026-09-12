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
//!
//! The layouts and the type codes below follow `H5O__shared_decode` in the C
//! library, which is the authority on what a file may contain: version 1 carries
//! a symbol-table entry (so its object-header address sits past a local-heap
//! address), version 2 and 3 carry the address or heap id directly, and the type
//! byte distinguishes the two destinations only from version 2 on.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use crate::address::BaseAddress;
use crate::bytes::{ensure_len, read_offset};
use crate::convert::TryToUsize;
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::sohm::SohmTable;
use crate::source::Source;

/// Fractal heap ID length for SOHM entries (fixed at 8 bytes).
pub(crate) const FHEAP_ID_LEN: usize = 8;

/// Shared-message location type: the message lives in the SOHM heap
/// (`H5O_SHARE_TYPE_SOHM`). Every other code names an object header, which is how
/// the C library reads them: `H5O__shared_decode` branches on this one value and
/// decodes an address for all the rest.
const REF_TYPE_SOHM: u8 = 1;

/// Shared-message location type: the message lives in another object header
/// (`H5O_SHARE_TYPE_COMMITTED`) — a committed datatype. This is the only type the
/// C library's encoder writes besides [`REF_TYPE_SOHM`], and the type a version 1
/// reference is defined to have.
const REF_TYPE_COMMITTED: u8 = 2;

/// Where a shared message actually lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SharedLocation {
    /// In another object header, at this address. A committed (`H5Tcommit`)
    /// datatype is stored this way, as is every version 1 reference.
    ObjectHeader(u64),
    /// In the file's shared object header message heap, under this fractal-heap
    /// id. Written only when a file enables SOHM indexes (`H5Pset_shared_mesg_*`).
    SohmHeap([u8; FHEAP_ID_LEN]),
}

/// A parsed shared message reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedMessageRef {
    /// Version of the shared message encoding (1, 2, or 3).
    pub version: u8,
    /// The raw location-type byte, as stored. Version 1 has no meaningful one and
    /// reports [`REF_TYPE_COMMITTED`], matching how the C library decodes it.
    pub ref_type: u8,
    /// Where the referenced message lives.
    pub location: SharedLocation,
}

/// Parse a shared message reference from a message body.
///
/// `length_size` is needed for version 1 only, whose reference is a symbol-table
/// entry: the object-header address follows a local-heap address of that width.
pub fn parse_shared_ref(
    data: &[u8],
    offset_size: u8,
    length_size: u8,
) -> Result<SharedMessageRef, FormatError> {
    ensure_len(data, 0, 2)?;
    let version = data[0];

    match version {
        1 => {
            // version(1) + unused type byte(1) + reserved(6) + a symbol-table
            // entry, whose local-heap address is skipped and whose object-header
            // address follows. Version 1 predates the SOHM table, so the type
            // byte carries nothing and the destination is always an object
            // header.
            let pos = 2 + 6 + length_size as usize;
            let addr = read_offset(data, pos, offset_size)?;
            Ok(SharedMessageRef {
                version,
                ref_type: REF_TYPE_COMMITTED,
                location: SharedLocation::ObjectHeader(addr),
            })
        }
        2 | 3 => {
            // version(1) + type(1) + either an 8-byte fractal-heap id (SOHM) or
            // an address. Version 2 has no reserved bytes: the C library skips
            // those for version 1 alone.
            let ref_type = data[1];
            let location = if ref_type == REF_TYPE_SOHM {
                ensure_len(data, 2, FHEAP_ID_LEN)?;
                let mut id = [0u8; FHEAP_ID_LEN];
                id.copy_from_slice(&data[2..2 + FHEAP_ID_LEN]);
                SharedLocation::SohmHeap(id)
            } else {
                SharedLocation::ObjectHeader(read_offset(data, 2, offset_size)?)
            };
            Ok(SharedMessageRef {
                version,
                ref_type,
                location,
            })
        }
        _ => Err(FormatError::InvalidSharedMessageVersion(version)),
    }
}

/// The shared-reference version this crate writes.
///
/// Version 2 is what libhdf5 1.14 encodes for every committed datatype, and the
/// only version whose body is just the address: version 1 buries it behind a
/// symbol-table entry, and version 3 differs only in admitting a heap id this
/// crate does not write.
const WRITE_REF_VERSION: u8 = 2;

/// Encode a reference to the committed datatype object at `address`.
///
/// The inverse of the version 2 arm of [`parse_shared_ref`], and the body a
/// message record with the shared flag carries in place of its content.
pub fn encode_committed_ref(address: u64, offset_size: u8) -> Vec<u8> {
    let mut buf = Vec::with_capacity(2 + offset_size as usize);
    buf.push(WRITE_REF_VERSION);
    buf.push(REF_TYPE_COMMITTED);
    buf.extend_from_slice(&address.to_le_bytes()[..offset_size as usize]);
    buf
}

/// The shared-reference version that admits a heap ID.
///
/// A message stored in the shared-message heap has no encoding before version 3:
/// versions 1 and 2 carry an object-header address and nothing else.
const SOHM_REF_VERSION: u8 = 3;

/// Encode a reference to the shared-message heap object `heap_id`.
///
/// The inverse of [`parse_shared_ref`]'s heap arm, and the modern form of a
/// reference a rewrite has to carry across unchanged: it names the same heap
/// entry, so the message's reference count is the same before and after.
pub fn encode_sohm_ref(heap_id: &[u8; FHEAP_ID_LEN]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(2 + FHEAP_ID_LEN);
    buf.push(SOHM_REF_VERSION);
    buf.push(REF_TYPE_SOHM);
    buf.extend_from_slice(heap_id);
    buf
}

/// Where a datatype is stored, for a message that could hold it either way.
///
/// A datatype is the one part of a dataset or attribute that can live outside
/// the message describing it: `H5Tcommit` puts it in its own object header, and
/// everything using it carries a reference in place of the encoding. Both forms
/// decode to the same [`Datatype`](crate::datatype::Datatype), so this is what
/// separates a message that *names* a type from one that spells it out — a
/// distinction `h5dump` reports, and a rewrite has to preserve.
///
/// No `Default`: an omitted location silently means `Inline`, and a reference
/// that decodes as an encoding is the whole defect this type exists to prevent.
/// Every construction names its variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatatypeLocation {
    /// Encoded in the message itself.
    Inline,
    /// A reference to the committed datatype object at this address, in the file
    /// the message belongs to. What a parse reads out of a file, and what a
    /// writer emits once the object's address is fixed.
    Committed(u64),
    /// Staged for writing: a reference to the committed datatype object the file
    /// under construction places at this path.
    ///
    /// Addresses are not known until the whole layout is, so the writer sizes
    /// headers against this variant and resolves it to [`Self::Committed`] in the
    /// same pass that assigns addresses. Serializing one writes the undefined
    /// address, so a reference that misses that pass names nothing rather than
    /// silently naming the superblock.
    CommittedPath(String),
}

impl DatatypeLocation {
    /// The reference body to write in place of the datatype encoding, or `None`
    /// when the datatype is written inline.
    pub fn reference_bytes(&self, offset_size: u8) -> Option<Vec<u8>> {
        match self {
            Self::Inline => None,
            Self::Committed(addr) => Some(encode_committed_ref(*addr, offset_size)),
            Self::CommittedPath(_) => Some(encode_committed_ref(u64::MAX, offset_size)),
        }
    }

    /// The path this location still has to have resolved, if any.
    pub fn unresolved_path(&self) -> Option<&str> {
        match self {
            Self::CommittedPath(path) => Some(path),
            Self::Inline | Self::Committed(_) => None,
        }
    }

    /// Whether the datatype lives in a committed object rather than in the
    /// message.
    pub fn is_committed(&self) -> bool {
        !matches!(self, Self::Inline)
    }
}

/// Reads the message a reference stands in for.
///
/// A reference names an address in the file, which the body holding it does not
/// carry, and the two reader backends reach the file differently — one indexes a
/// slice, the other reads a [`Source`]. Parsers that may meet a reference take
/// one of these rather than either backend directly.
pub trait SharedResolver {
    /// Resolve `reference` — the body of a shared message — into the bytes of the
    /// `target`-typed message it names.
    fn resolve(&self, reference: &[u8], target: MessageType) -> Result<Vec<u8>, FormatError>;

    /// The object-header address `reference` names, without reading it, or
    /// `None` for a reference into the shared-message heap.
    ///
    /// A rewrite needs the address as well as the content: the content says what
    /// the type *is*, and the address says which committed object every user of
    /// it shares — which is what makes them one named type on the other side
    /// rather than several copies. A heap-stored message has no such object: it
    /// is one copy of an *anonymous* message, which every user spells out again
    /// when written back, so `None` is the answer rather than an error.
    fn committed_address(&self, reference: &[u8]) -> Result<Option<u64>, FormatError>;
}

/// Resolves references against a whole-file slice, already framed at the file's
/// base address (shared-message addresses are stored relative to it).
pub struct BufferedResolver<'a> {
    file_data: &'a [u8],
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
        offset_size: u8,
        length_size: u8,
        sohm: Option<&'a SohmTable>,
    ) -> Self {
        Self {
            file_data,
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
            addr.to_usize()?,
            self.offset_size,
            self.length_size,
        )?;
        select_shared_message(&header, target, addr)
    }

    fn committed_address(&self, reference: &[u8]) -> Result<Option<u64>, FormatError> {
        committed_address_in(reference, self.offset_size, self.length_size)
    }
}

/// Resolves references by reading the target object header from a [`Source`] on
/// demand instead of indexing a whole-file slice.
pub struct SourceResolver<'a, S: Source + ?Sized> {
    source: &'a S,
    offset_size: u8,
    length_size: u8,
    sohm: Option<&'a SohmTable>,
}

impl<'a, S: Source + ?Sized> SourceResolver<'a, S> {
    /// See [`BufferedResolver::new`] for why the shared-message table is a
    /// parameter here rather than something the resolver finds for itself.
    pub fn new(
        source: &'a S,
        offset_size: u8,
        length_size: u8,
        sohm: Option<&'a SohmTable>,
    ) -> Self {
        Self {
            source,
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
            self.source,
            addr,
            self.offset_size,
            self.length_size,
            BaseAddress::ZERO,
        )?;
        select_shared_message(&header, target, addr)
    }

    fn committed_address(&self, reference: &[u8]) -> Result<Option<u64>, FormatError> {
        committed_address_in(reference, self.offset_size, self.length_size)
    }
}

/// Refuses every reference, for parses that hold a message body but not the file
/// it came from. Returning the encoding stored *at* the reference would be a
/// different message entirely, so the only honest answer is an error.
pub struct Unresolvable;

impl SharedResolver for Unresolvable {
    fn resolve(&self, _reference: &[u8], target: MessageType) -> Result<Vec<u8>, FormatError> {
        Err(FormatError::UnresolvedSharedMessage(target.to_u16()))
    }

    /// Refused for the same reason as [`Self::resolve`]: the address is stored in
    /// the file's own offset width, which a parse without that file does not
    /// know, so any answer here would be a guess at the field width.
    fn committed_address(&self, _reference: &[u8]) -> Result<Option<u64>, FormatError> {
        Err(FormatError::UnresolvedSharedMessage(
            MessageType::Datatype.to_u16(),
        ))
    }
}

/// The object-header address `reference` names, read with the given field widths,
/// or `None` for a reference into the shared-message heap.
///
/// Every resolver that can reach the file answers
/// [`SharedResolver::committed_address`] this way; they differ only in how they
/// read the object *at* the address, which is [`SharedResolver::resolve`]'s job.
pub(crate) fn committed_address_in(
    reference: &[u8],
    offset_size: u8,
    length_size: u8,
) -> Result<Option<u64>, FormatError> {
    match parse_shared_ref(reference, offset_size, length_size)?.location {
        SharedLocation::ObjectHeader(addr) => Ok(Some(addr)),
        SharedLocation::SohmHeap(_) => Ok(None),
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

    /// Version 2 is version + type + address, with no reserved bytes. This is the
    /// encoding libhdf5 writes for every committed datatype, so reading the
    /// address anywhere else lands in whatever follows the field.
    #[test]
    fn parse_v2_committed_ref() {
        let mut data = vec![2, REF_TYPE_COMMITTED];
        data.extend_from_slice(&0x320u64.to_le_bytes());

        let shared = parse_shared_ref(&data, 8, 8).unwrap();
        assert_eq!(shared.version, 2);
        assert_eq!(shared.ref_type, REF_TYPE_COMMITTED);
        assert_eq!(shared.location, SharedLocation::ObjectHeader(0x320));
    }

    /// The exact 10-byte field h5py 3.14 / libhdf5 1.14.6 wrote for an attribute
    /// whose datatype is `f["mytype"]`, address and all. A layout change that
    /// still parses would move the address, so the value is the assertion.
    #[test]
    fn parse_v2_ref_as_libhdf5_writes_it() {
        let data = [0x02, 0x02, 0x20, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let shared = parse_shared_ref(&data, 8, 8).unwrap();
        assert_eq!(shared.location, SharedLocation::ObjectHeader(800));
    }

    /// Version 1 stores a symbol-table entry: the object-header address follows a
    /// local-heap address of `length_size` bytes, not the reserved bytes alone.
    #[test]
    fn parse_v1_ref_skips_the_local_heap_address() {
        let mut data = vec![1, 0];
        data.extend_from_slice(&[0u8; 6]); // reserved
        data.extend_from_slice(&0x1111u64.to_le_bytes()); // local heap address
        data.extend_from_slice(&0x5678u64.to_le_bytes()); // object header address

        let shared = parse_shared_ref(&data, 8, 8).unwrap();
        assert_eq!(shared.version, 1);
        assert_eq!(shared.location, SharedLocation::ObjectHeader(0x5678));
    }

    /// A version 1 reference in a file with 4-byte lengths puts the address four
    /// bytes earlier, so the skip is the file's length size and not a constant.
    #[test]
    fn parse_v1_ref_uses_the_files_length_size() {
        let mut data = vec![1, 0];
        data.extend_from_slice(&[0u8; 6]);
        data.extend_from_slice(&0x1111u32.to_le_bytes()); // local heap address
        data.extend_from_slice(&0x5678u32.to_le_bytes()); // object header address

        let shared = parse_shared_ref(&data, 4, 4).unwrap();
        assert_eq!(shared.location, SharedLocation::ObjectHeader(0x5678));
    }

    #[test]
    fn parse_v3_committed_ref() {
        let mut data = vec![3, REF_TYPE_COMMITTED];
        data.extend_from_slice(&0xABCDu64.to_le_bytes());

        let shared = parse_shared_ref(&data, 8, 8).unwrap();
        assert_eq!(shared.version, 3);
        assert_eq!(shared.location, SharedLocation::ObjectHeader(0xABCD));
    }

    /// Type 1 is the SOHM heap, not an object header. Reading its 8-byte heap id
    /// as an address is how a fractal-heap id becomes a plausible file offset.
    #[test]
    fn parse_v3_sohm_ref() {
        let mut data = vec![3, REF_TYPE_SOHM];
        data.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22, 0x33, 0x44]);

        let shared = parse_shared_ref(&data, 8, 8).unwrap();
        assert_eq!(shared.ref_type, REF_TYPE_SOHM);
        assert_eq!(
            shared.location,
            SharedLocation::SohmHeap([0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22, 0x33, 0x44])
        );
    }

    /// A heap reference re-encoded from a parse is the same eight bytes at the
    /// same offset, under the only version that carries them.
    #[test]
    fn a_heap_reference_round_trips_through_its_encoding() {
        let id = [0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22, 0x33, 0x44];
        let encoded = encode_sohm_ref(&id);
        assert_eq!(encoded[0], 3);
        assert_eq!(
            parse_shared_ref(&encoded, 8, 8).unwrap().location,
            SharedLocation::SohmHeap(id)
        );
    }

    #[test]
    fn parse_v3_sohm_too_short() {
        let data = vec![3, REF_TYPE_SOHM, 0xAA, 0xBB];
        let err = parse_shared_ref(&data, 8, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn invalid_version() {
        let data = vec![99, 0];
        let err = parse_shared_ref(&data, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidSharedMessageVersion(99));
    }

    #[test]
    fn truncated_data() {
        let data = vec![3u8]; // too short
        let err = parse_shared_ref(&data, 8, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn parse_four_byte_offsets() {
        let mut data = vec![3, REF_TYPE_COMMITTED];
        data.extend_from_slice(&0x1000u32.to_le_bytes());

        let shared = parse_shared_ref(&data, 4, 4).unwrap();
        assert_eq!(shared.location, SharedLocation::ObjectHeader(0x1000));
    }

    /// A resolver given no shared-message table refuses a heap reference by
    /// name. Its heap id is not an address, so the alternative is an
    /// object-header parse at whatever those eight bytes spell.
    #[test]
    fn a_sohm_reference_without_a_table_is_refused_rather_than_followed() {
        let mut reference = vec![3, REF_TYPE_SOHM];
        reference.extend_from_slice(&[0xFF; 8]);
        let resolver = BufferedResolver::new(&[], 8, 8, None);

        let err = resolver
            .resolve(&reference, MessageType::Datatype)
            .unwrap_err();
        assert_eq!(err, FormatError::UnsupportedSohmReference);
    }

    /// A heap reference names no object header, and says so with `None` rather
    /// than an error: the message it stands for is one anonymous copy, not a
    /// committed object every user shares by name.
    #[test]
    fn a_sohm_reference_names_no_committed_object() {
        let mut reference = vec![3, REF_TYPE_SOHM];
        reference.extend_from_slice(&[0xFF; 8]);
        assert_eq!(committed_address_in(&reference, 8, 8).unwrap(), None);
    }

    /// The target header must hold the message the reference stands in for.
    #[test]
    fn a_reference_to_a_header_without_that_message_is_an_error() {
        let header = header_with(vec![message(
            MessageType::Dataspace,
            MessageFlags::NONE,
            vec![1, 2, 3],
        )]);
        let err = select_shared_message(&header, MessageType::Datatype, 0x320).unwrap_err();
        assert_eq!(
            err,
            FormatError::SharedMessageMissing {
                object_header_address: 0x320,
                message_type: MessageType::Datatype.to_u16(),
            }
        );
    }

    /// A message that is itself a reference is not content, so it is not an
    /// answer: handing its bytes back would re-create the mis-decode one level
    /// down.
    #[test]
    fn a_shared_message_in_the_target_is_not_mistaken_for_content() {
        let header = header_with(vec![message(
            MessageType::Datatype,
            MessageFlags::SHARED,
            vec![2, 2, 0, 0],
        )]);
        let err = select_shared_message(&header, MessageType::Datatype, 0x320).unwrap_err();
        assert!(matches!(err, FormatError::SharedMessageMissing { .. }));
    }

    /// Nothing resolves without the file the reference addresses.
    #[test]
    fn the_unresolvable_resolver_refuses() {
        let err = Unresolvable
            .resolve(
                &[2, REF_TYPE_COMMITTED, 0, 0, 0, 0, 0, 0, 0, 0],
                MessageType::Datatype,
            )
            .unwrap_err();
        assert_eq!(
            err,
            FormatError::UnresolvedSharedMessage(MessageType::Datatype.to_u16())
        );
    }
}
