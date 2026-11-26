use alloc::vec::Vec;

use super::raw::NAME_HASH_LEN;
use crate::btree::v2::BTreeV2Record;
use crate::bytes;
use crate::error::FormatError;
use crate::fractal_heap::{FractalHeapIdLayout, FractalHeapIdView};
use crate::message_flags::MessageFlags;
use crate::shared_message::FHEAP_ID_LEN;

impl BTreeV2Record {
    /// Returns the heap ID that opens a type 8 or type 9 record of an attribute index, or `None`
    /// if the record is shorter than `heap_id_length` bytes.
    pub fn attribute_heap_id(&self, heap_id_length: u16) -> Option<&[u8]> {
        self.data.get(..usize::from(heap_id_length))
    }

    /// Returns the creation order in a type 8 or type 9 record of an attribute index, past a heap
    /// ID of `heap_id_length` bytes and the message flags, or `None` if the record ends before it.
    pub fn attribute_creation_order(&self, heap_id_length: u16) -> Option<u32> {
        let at = usize::from(heap_id_length) + usize::from(MESSAGE_FLAGS_LEN);
        self.data
            .get(at..)
            .and_then(|rest| rest.first_chunk())
            .map(|&bytes| u32::from_le_bytes(bytes))
    }
}

/// A record of a version 2 B-tree that indexes the attributes of an object in dense storage, type
/// 8 in the name index and type 9 in the creation-order index.
///
/// A type 9 record stores the three fields, and a type 8 record adds the hash of the attribute
/// name. Both are defined in "Version 2 B-trees" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttributeRecord<'a> {
    /// The heap ID of the Attribute message. The [`flags`](Self::flags) field specifies the
    /// file's shared-message heap for a shared message, or the object's attribute heap otherwise.
    pub heap_id: &'a [u8],
    /// The object header message flags of the Attribute message.
    pub flags: MessageFlags,
    /// The creation order of the attribute.
    pub creation_order: u32,
}

impl AttributeRecord<'_> {
    /// Returns the length in bytes of a type 8 record whose heap ID is `heap_id_length` bytes
    /// long.
    pub const fn name_record_size(heap_id_length: u16) -> u16 {
        Self::creation_order_record_size(heap_id_length) + NAME_HASH_LEN
    }

    /// Returns the length in bytes of a type 9 record whose heap ID is `heap_id_length` bytes
    /// long.
    pub const fn creation_order_record_size(heap_id_length: u16) -> u16 {
        heap_id_length + MESSAGE_FLAGS_LEN + ATTRIBUTE_CREATION_ORDER_LEN
    }

    /// Appends the type 8 record of the attribute to `buf`, with `name_hash` as the hash of its
    /// name.
    pub fn encode_name_record(&self, buf: &mut Vec<u8>, name_hash: u32) {
        self.encode_creation_order_record(buf);
        buf.extend_from_slice(&name_hash.to_le_bytes());
    }

    /// Appends the type 9 record of the attribute to `buf`.
    pub fn encode_creation_order_record(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(self.heap_id);
        buf.push(self.flags.get());
        buf.extend_from_slice(&self.creation_order.to_le_bytes());
    }
}

/// The common fields of a dense attribute name or creation-order record.
///
/// The record borrows the heap ID bytes. Call [`target`](Self::target) to select the
/// private or shared-message heap using the message flags.
///
/// Types 8 and 9 are defined in "Version 2 B-trees" of the [format specification, version
/// 4.0][spec]. libhdf5 selects the shared-message heap when the shared flag is set
/// (`H5A__dense_btree2_name_compare` in `H5Abtree2.c`, HDF5 2.2.0).
///
/// # Examples
///
/// ```
/// # use hdf5_pure_format::__private::{
/// #     AttributeRecord, BTREE_V2_ATTRIBUTE_NAME, DenseAttributeRecord, DenseAttributeTarget,
/// #     FractalHeapHeader, FractalHeapIdLayout, LengthWidth, MessageFlags, OffsetWidth,
/// # };
/// # use hdf5_pure_format::FormatError;
/// # use test_util::{fractal_heap, widths::Widths};
/// # let mut heap = FractalHeapHeader::parse_bytes(
/// #     &fractal_heap::Header::new(256).build(Widths::EIGHT),
/// #     OffsetWidth::Eight,
/// #     LengthWidth::Eight,
/// # )?;
/// # heap.heap_id_length = 8;
/// # let layout = FractalHeapIdLayout::new(&heap, OffsetWidth::Eight, LengthWidth::Eight);
/// let mut bytes = Vec::new();
/// AttributeRecord {
///     heap_id: &[0xFF; 8],
///     flags: MessageFlags::SHARED,
///     creation_order: 42,
/// }
/// .encode_name_record(&mut bytes, 0);
/// let record = DenseAttributeRecord::parse(&bytes, BTREE_V2_ATTRIBUTE_NAME, 8)?;
/// assert_eq!(record.flags(), MessageFlags::SHARED);
/// let DenseAttributeTarget::Shared(target) = record.target(&layout)? else {
///     panic!("expected a shared target");
/// };
/// assert_eq!(target.bytes(), &[0xFF; 8]);
/// # Ok::<(), FormatError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseAttributeRecord<'a> {
    heap_id: &'a [u8],
    flags: MessageFlags,
    creation_order: u32,
}

impl<'a> DenseAttributeRecord<'a> {
    /// Reads a type 8 or type 9 record with the owner's `heap_id_length`.
    ///
    /// A type 8 record includes a trailing name hash, which is checked for completeness and
    /// omitted from the parsed fields.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeNodeType`] if `tree_type` is not 8 or 9, and
    /// [`FormatError::UnexpectedEof`] if the record is truncated.
    pub fn parse(data: &'a [u8], tree_type: u8, heap_id_length: u16) -> Result<Self, FormatError> {
        let suffix = match tree_type {
            BTREE_V2_ATTRIBUTE_NAME => usize::from(NAME_HASH_LEN),
            BTREE_V2_ATTRIBUTE_CREATION_ORDER => 0,
            other => return Err(FormatError::InvalidBTreeNodeType(other)),
        };
        let id_len = usize::from(heap_id_length);
        let required = id_len
            + usize::from(MESSAGE_FLAGS_LEN)
            + usize::from(ATTRIBUTE_CREATION_ORDER_LEN)
            + suffix;
        bytes::ensure_len(data, 0, required)?;
        let heap_id = data.get(..id_len).ok_or(FormatError::UnexpectedEof {
            expected: id_len,
            available: data.len(),
        })?;
        let mut cursor = bytes::ByteCursor::with_position(data, id_len)?;
        Ok(Self {
            heap_id,
            flags: MessageFlags::new(cursor.u8()?),
            creation_order: cursor.u32()?,
        })
    }

    /// Returns the Attribute message flags used to select its heap.
    pub const fn flags(self) -> MessageFlags {
        self.flags
    }

    /// Returns the attribute creation order stored in the record.
    pub const fn creation_order(self) -> u32 {
        self.creation_order
    }

    /// Returns the attribute target for the heap selected by the message flags.
    ///
    /// Private IDs are parsed with `private_layout`. Shared IDs are copied into an array of
    /// [`FHEAP_ID_LEN`] bytes for the shared-message heap reader.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapIdLength`] if a shared ID is not
    /// [`FHEAP_ID_LEN`] bytes long, or the error from [`FractalHeapIdLayout::parse`] for an
    /// invalid private ID.
    pub fn target(
        self,
        private_layout: &FractalHeapIdLayout,
    ) -> Result<DenseAttributeTarget<'a>, FormatError> {
        Ok(if self.flags.is_shared() {
            DenseAttributeTarget::Shared(SharedAttributeId {
                bytes: self.heap_id.try_into().map_err(|_error| {
                    FormatError::InvalidFractalHeapIdLength {
                        expected: FHEAP_ID_LEN,
                        actual: self.heap_id.len(),
                    }
                })?,
            })
        } else {
            DenseAttributeTarget::Private(PrivateAttributeId {
                id: private_layout.parse(self.heap_id)?,
            })
        })
    }
}

/// A parsed private attribute ID or the eight bytes of a shared attribute ID.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseAttributeTarget<'a> {
    /// An Attribute message in the object's attribute heap.
    Private(PrivateAttributeId<'a>),
    /// An Attribute message in the file's shared-message heap.
    Shared(SharedAttributeId),
}

/// An attribute ID parsed against the owner's [`FractalHeapIdLayout`].
///
/// Constructed by [`DenseAttributeRecord::target`] for a private record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrivateAttributeId<'a> {
    id: FractalHeapIdView<'a>,
}

impl<'a> PrivateAttributeId<'a> {
    /// Returns the parsed ID for the owner's heap reader.
    pub const fn id(self) -> FractalHeapIdView<'a> {
        self.id
    }
}

/// An eight-byte attribute ID for the file's shared-message heap.
///
/// Constructed by [`DenseAttributeRecord::target`] for a shared record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SharedAttributeId {
    bytes: [u8; FHEAP_ID_LEN],
}

impl SharedAttributeId {
    /// Returns the raw heap ID for the shared-message heap reader.
    pub const fn bytes(&self) -> &[u8; FHEAP_ID_LEN] {
        &self.bytes
    }
}

/// The type of a version 2 B-tree that indexes the attribute names of an object, from the same
/// table as [`BTREE_V2_HUGE_OBJECT`](super::BTREE_V2_HUGE_OBJECT).
pub const BTREE_V2_ATTRIBUTE_NAME: u8 = 8;

/// The type of a version 2 B-tree that indexes the attribute creation order of an object, from
/// the same table as [`BTREE_V2_HUGE_OBJECT`](super::BTREE_V2_HUGE_OBJECT).
pub const BTREE_V2_ATTRIBUTE_CREATION_ORDER: u8 = 9;

/// The width of the message flags in a type 8 or type 9 record.
const MESSAGE_FLAGS_LEN: u16 = 1;

/// The width of the creation order in a type 8 or type 9 record.
const ATTRIBUTE_CREATION_ORDER_LEN: u16 = 4;

#[cfg(test)]
#[path = "tests/attribute.rs"]
mod tests;
