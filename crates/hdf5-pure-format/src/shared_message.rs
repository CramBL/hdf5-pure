//! The shared message reference, the body a message record holds in place of a shared message.
//!
//! The reference is defined in "Data Object Header Messages" of the [format specification, version
//! 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_dataobject_hdr_msg

use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::bytes;
use crate::bytes::{ensure_len, read_offset};
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::width::OffsetWidth;

/// Fractal heap ID length for SOHM entries (fixed at 8 bytes).
pub const FHEAP_ID_LEN: usize = 8;

/// Shared-message location type: the message lives in the SOHM heap
/// (`H5O_SHARE_TYPE_SOHM`). Every other code identifies an object header, which is how
/// the C library reads them: `H5O__shared_decode` branches on this one value and
/// decodes an address for all the rest.
const REF_TYPE_SOHM: u8 = 1;

/// Shared-message location type: the message lives in another object header
/// (`H5O_SHARE_TYPE_COMMITTED`), a committed datatype. This is the only type the
/// C library's encoder writes besides [`REF_TYPE_SOHM`], and the type a version 1
/// reference is defined to have.
const REF_TYPE_COMMITTED: u8 = 2;

/// Where a shared message actually lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SharedLocation {
    /// In another object header, at this address. A committed (`H5Tcommit`)
    /// datatype is stored this way, as is every version 1 reference.
    ObjectHeader(StoredAddress),
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
    /// reports `REF_TYPE_COMMITTED`, matching how the C library decodes it.
    pub ref_type: u8,
    /// Where the referenced message lives.
    pub location: SharedLocation,
}

/// Parses a shared message reference from a message body.
///
/// `length_size` is needed for version 1 only, whose reference is a symbol-table
/// entry: the object-header address follows a local-heap address of that width.
///
/// # Errors
///
/// Returns [`FormatError::InvalidSharedMessageVersion`] if the version is not 1, 2, or 3,
/// [`FormatError::InvalidOffsetSize`] if the reference holds an address and `offset_size` is not
/// 2, 4, or 8, and [`FormatError::UnexpectedEof`] if `data` ends inside the reference.
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
            // byte holds nothing and the destination is always an object
            // header.
            let pos = 2 + 6 + length_size as usize;
            let addr = StoredAddress::new(read_offset(data, pos, offset_size)?);
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
                SharedLocation::ObjectHeader(StoredAddress::new(read_offset(data, 2, offset_size)?))
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

/// Encodes a reference to the committed datatype object at `address`.
///
/// The inverse of the version 2 arm of [`parse_shared_ref`], and the body a
/// message record with the shared flag holds in place of its content.
///
/// # Panics
///
/// Panics if `address` does not fit `offset_width`.
pub fn encode_committed_ref(address: StoredAddress, offset_width: OffsetWidth) -> Vec<u8> {
    let mut buf = Vec::with_capacity(2 + usize::from(offset_width.get()));
    buf.push(WRITE_REF_VERSION);
    buf.push(REF_TYPE_COMMITTED);
    bytes::write_offset(&mut buf, address.get(), offset_width);
    buf
}

/// The shared-reference version that admits a heap ID.
///
/// A message stored in the shared-message heap has no encoding before version 3:
/// versions 1 and 2 define an object-header address and nothing else.
const SOHM_REF_VERSION: u8 = 3;

/// Encodes a reference to the shared-message heap object `heap_id`.
///
/// The inverse of [`parse_shared_ref`]'s heap arm, and the modern form of a
/// reference a rewrite has to keep unchanged: it refers to the same heap
/// entry, so the message's reference count is the same before and after.
pub fn encode_sohm_ref(heap_id: &[u8; FHEAP_ID_LEN]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(2 + FHEAP_ID_LEN);
    buf.push(SOHM_REF_VERSION);
    buf.push(REF_TYPE_SOHM);
    buf.extend_from_slice(heap_id);
    buf
}

/// Reads the message that a shared message reference stands for.
///
/// A reference locates the message in another object header or in the shared message heap, so
/// resolving it reads past the message body. A parser that may meet a reference takes a resolver,
/// and a caller that has a message body without its file passes [`Unresolvable`].
pub trait SharedResolver {
    /// Resolves `reference`, the body of a shared message, into the bytes of the
    /// `target`-typed message it refers to.
    ///
    /// # Errors
    ///
    /// An implementation returns an error if `reference` is malformed or the read of the message
    /// fails. [`Unresolvable`] returns [`FormatError::UnresolvedSharedMessage`] for every
    /// reference.
    fn resolve(&self, reference: &[u8], target: MessageType) -> Result<Vec<u8>, FormatError>;

    /// Returns the object header address `reference` names, without reading the object there,
    /// or `None` for a reference into the shared-message heap.
    ///
    /// A rewrite needs the address as well as the content: the content says what
    /// the type *is*, and the address says which committed object every user of
    /// it shares, which is what makes them one named type on the other side
    /// and not several copies. A heap-stored message has no such object: it
    /// is one copy of an *anonymous* message, which every user spells out again
    /// when written back, so the result is `None` and not an error.
    ///
    /// # Errors
    ///
    /// An implementation returns an error if `reference` is malformed. [`Unresolvable`] returns
    /// [`FormatError::UnresolvedSharedMessage`] for every reference.
    fn committed_address(&self, reference: &[u8]) -> Result<Option<StoredAddress>, FormatError>;
}

/// Rejects every reference, for parses that hold a message body but not the file
/// it came from. Returning the encoding stored *at* the reference would be a
/// different message entirely, so the only honest result is an error.
pub struct Unresolvable;

impl SharedResolver for Unresolvable {
    fn resolve(&self, _reference: &[u8], target: MessageType) -> Result<Vec<u8>, FormatError> {
        Err(FormatError::UnresolvedSharedMessage(target.to_u16()))
    }

    /// Rejected for the same reason as [`Self::resolve`]: the address is stored in
    /// the file's own offset width, which a parse without that file does not
    /// know, so any result here would be a guess at the field width.
    fn committed_address(&self, _reference: &[u8]) -> Result<Option<StoredAddress>, FormatError> {
        Err(FormatError::UnresolvedSharedMessage(
            MessageType::DATATYPE.to_u16(),
        ))
    }
}

/// Returns the object header address `reference` refers to, read with `offset_size` and
/// `length_size`, or `None` for a reference into the shared-message heap.
///
/// Every resolver that can reach the file implements
/// [`SharedResolver::committed_address`] this way. They differ only in how they
/// read the object *at* the address, which is [`SharedResolver::resolve`]'s job.
///
/// # Errors
///
/// Returns the error [`parse_shared_ref`] returns if `reference` is malformed.
pub fn committed_address_in(
    reference: &[u8],
    offset_size: u8,
    length_size: u8,
) -> Result<Option<StoredAddress>, FormatError> {
    match parse_shared_ref(reference, offset_size, length_size)?.location {
        SharedLocation::ObjectHeader(addr) => Ok(Some(addr)),
        SharedLocation::SohmHeap(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            shared.location,
            SharedLocation::ObjectHeader(StoredAddress::new(0x320))
        );
    }

    /// The exact 10-byte field h5py 3.14 / libhdf5 1.14.6 wrote for an attribute
    /// whose datatype is `f["mytype"]`, address and all. A layout change that
    /// still parses would move the address, so the value is the assertion.
    #[test]
    fn parse_v2_ref_as_libhdf5_writes_it() {
        let data = [0x02, 0x02, 0x20, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let shared = parse_shared_ref(&data, 8, 8).unwrap();
        assert_eq!(
            shared.location,
            SharedLocation::ObjectHeader(StoredAddress::new(800))
        );
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
        assert_eq!(
            shared.location,
            SharedLocation::ObjectHeader(StoredAddress::new(0x5678))
        );
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
        assert_eq!(
            shared.location,
            SharedLocation::ObjectHeader(StoredAddress::new(0x5678))
        );
    }

    #[test]
    fn parse_v3_committed_ref() {
        let mut data = vec![3, REF_TYPE_COMMITTED];
        data.extend_from_slice(&0xABCDu64.to_le_bytes());

        let shared = parse_shared_ref(&data, 8, 8).unwrap();
        assert_eq!(shared.version, 3);
        assert_eq!(
            shared.location,
            SharedLocation::ObjectHeader(StoredAddress::new(0xABCD))
        );
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
    /// same offset, under version 3, the only version that defines a heap ID.
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
        assert_eq!(
            shared.location,
            SharedLocation::ObjectHeader(StoredAddress::new(0x1000))
        );
    }

    /// A heap reference identifies no object header, and says so with `None`, not
    /// an error: the message it stands for is one anonymous copy, not a
    /// committed object every user shares by name.
    #[test]
    fn a_sohm_reference_names_no_committed_object() {
        let mut reference = vec![3, REF_TYPE_SOHM];
        reference.extend_from_slice(&[0xFF; 8]);
        assert_eq!(committed_address_in(&reference, 8, 8).unwrap(), None);
    }

    /// Nothing resolves without the file the reference addresses.
    #[test]
    fn the_unresolvable_resolver_refuses() {
        let err = Unresolvable
            .resolve(
                &[2, REF_TYPE_COMMITTED, 0, 0, 0, 0, 0, 0, 0, 0],
                MessageType::DATATYPE,
            )
            .unwrap_err();
        assert_eq!(
            err,
            FormatError::UnresolvedSharedMessage(MessageType::DATATYPE.to_u16())
        );
    }
}
