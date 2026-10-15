//! The global heap collections that the objects reachable from the root group refer to.

use alloc::collections::BTreeSet;
use core::num::NonZeroU32;
use core::num::NonZeroU64;

use crate::access_mode::AccessMode;
use crate::address::StoredAddress;
use crate::attribute::AttributeMessage;
use crate::attribute_info::AttributeInfoMessage;
use crate::convert::Narrow;
use crate::data_layout::DataLayout;
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::Error;
use crate::error::FormatError;
use crate::fill_value::FillPattern;
use crate::fill_value::FillValueView;
use crate::message_type::MessageType;
use crate::object_header::HeaderMessage;
use crate::object_header::ObjectHeader;
use crate::read_spec::RawReadSpec;
use crate::read_spec::RawReadStorage;
use crate::source::BaseOffsetSource;
use crate::vl_data;
use crate::vl_data::EmbeddedVlSlot;
use crate::vl_data::VlElement;

use super::WriteEngine;

/// The global heap collections that the objects reachable from the root group refer to.
///
/// The editor follows the hard links of each group and reads the variable-length elements of
/// each attribute, compact, contiguous, or chunked dataset, and fill value, including the
/// variable-length members of compound and array elements. The analysis is incomplete if it
/// reaches a shared message, an attribute with a shared datatype or dataspace, a virtual dataset,
/// a chunk it cannot read or decode, a datatype that holds a reference or a variable-length type
/// nested in another variable-length type, a datatype too small for its variable-length members,
/// a message of a type it does not recognize, or a traversal limit.
///
/// This type is outside the supported API.
#[doc(hidden)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlobalHeapLiveness {
    /// The absolute file offset of each collection, counted from byte 0 and including any
    /// userblock.
    pub referenced_collections: BTreeSet<u64>,
    /// Whether the editor read every reachable structure that can refer to a collection.
    ///
    /// A collection outside [`referenced_collections`](Self::referenced_collections) can still be
    /// live when this is `false`.
    pub analysis_complete: bool,
}

impl WriteEngine {
    /// Returns the global heap collections that the objects reachable from the root group refer
    /// to.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the metadata of a reachable object or one of its
    /// variable-length elements is malformed, and [`Error::MissingMessage`] if a dataset lacks its
    /// datatype or dataspace message.
    pub(crate) fn global_heap_liveness(&self) -> Result<GlobalHeapLiveness, Error> {
        let mut referenced_collections = BTreeSet::new();
        let root = self
            .addresses
            .absolute(self.superblock.root_group_address)?;
        let analysis_complete = self.walk_reachable_objects(root, |_, header, child| {
            self.collect_object_collections(header, child, &mut referenced_collections)
        })?;
        Ok(GlobalHeapLiveness {
            referenced_collections,
            analysis_complete,
        })
    }

    /// Adds the collections an object refers to, and passes the children of a group to `child`.
    ///
    /// Returns `false` if the object has a shared message, a message of a type the analysis does
    /// not recognize, or an attribute or a dataset that the analysis does not read in full.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`global_heap_liveness`](Self::global_heap_liveness).
    fn collect_object_collections(
        &self,
        header: &ObjectHeader,
        child: &mut dyn FnMut(u64),
        referenced: &mut BTreeSet<u64>,
    ) -> Result<bool, Error> {
        if crate::group_v2::is_group(header) {
            for entry in crate::group_v2::resolve_group_entries_from_source(
                &self.image(),
                AccessMode::ReadWrite,
                header,
                self.widths(),
                self.addresses,
            )? {
                if !entry
                    .object_header_address
                    .is_undefined(self.widths().offsets)
                {
                    child(self.addresses.absolute(entry.object_header_address)?);
                }
            }
        }
        let mut complete = true;
        for HeaderMessage {
            msg_type,
            flags,
            data,
            ..
        } in &header.messages
        {
            if flags.is_shared() {
                complete = false;
                continue;
            }
            match *msg_type {
                MessageType::ATTRIBUTE => {
                    complete &= self.collect_attribute_collections(data, referenced)?;
                }
                MessageType::ATTRIBUTE_INFO => {
                    complete &= self.collect_dense_attribute_collections(data, referenced)?;
                }
                MessageType::NIL
                | MessageType::DATASPACE
                | MessageType::LINK_INFO
                | MessageType::DATATYPE
                | MessageType::FILL_VALUE_OLD
                | MessageType::FILL_VALUE
                | MessageType::LINK
                | MessageType::DATA_LAYOUT
                | MessageType::GROUP_INFO
                | MessageType::FILTER_PIPELINE
                | MessageType::OBJECT_HEADER_CONTINUATION
                | MessageType::SYMBOL_TABLE
                | MessageType::OBJECT_MODIFICATION_TIME
                | MessageType::OBJECT_REFERENCE_COUNT => {}
                _ => complete = false,
            }
        }
        complete &= self.collect_dataset_collections(&header.messages, referenced)?;
        Ok(complete)
    }

    /// Adds the collections that a dataset's elements and fill value refer to.
    ///
    /// Returns `true` for an object without a data layout message, and `false` if the layout,
    /// datatype, dataspace, or filter pipeline message is shared or the layout is virtual. For a
    /// datatype that holds a file address, it also returns `false` if [`HeapReferences::of`]
    /// returns [`HeapReferences::Unsupported`] for the datatype, or if
    /// [`collect_chunked_collections`](Self::collect_chunked_collections) returns `false`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingMessage`] if the object has a data layout message and lacks its
    /// datatype or dataspace message, and [`Error::Format`] if a message, the chunk index, or an
    /// element is malformed.
    fn collect_dataset_collections(
        &self,
        messages: &[HeaderMessage],
        referenced: &mut BTreeSet<u64>,
    ) -> Result<bool, Error> {
        let find = |msg_type| messages.iter().find(|m| m.msg_type == msg_type);
        let Some(layout) = find(MessageType::DATA_LAYOUT) else {
            return Ok(true);
        };
        let datatype =
            find(MessageType::DATATYPE).ok_or(Error::MissingMessage(MessageType::DATATYPE))?;
        let dataspace =
            find(MessageType::DATASPACE).ok_or(Error::MissingMessage(MessageType::DATASPACE))?;
        if [layout, datatype, dataspace]
            .iter()
            .any(|m| m.flags.is_shared())
        {
            return Ok(false);
        }
        let pipeline = find(MessageType::FILTER_PIPELINE);
        if pipeline.is_some_and(|m| m.flags.is_shared()) {
            return Ok(false);
        }
        let layout = DataLayout::parse_with_widths(&layout.data, self.widths())?;
        let (datatype, _) = hdf5_pure_format::__private::parse_datatype(&datatype.data)?;
        if matches!(layout, DataLayout::Virtual) {
            return Ok(false);
        }
        let slots = match HeapReferences::of(&datatype)? {
            HeapReferences::Absent => return Ok(true),
            HeapReferences::Located(slots) => slots,
            HeapReferences::Unsupported => return Ok(false),
        };
        self.collect_fill_value_collections(messages, &slots, referenced)?;
        let dataspace = Dataspace::parse_with_length_width(&dataspace.data, self.widths().lengths)?;
        let pipeline = pipeline
            .map(|m| crate::filter_pipeline::parse_filter_pipeline(&m.data))
            .transpose()?;
        let spec = RawReadSpec::parse(
            &layout,
            &dataspace,
            &datatype,
            pipeline.as_ref(),
            FillPattern::ZERO,
        )?;
        Ok(match spec.storage() {
            RawReadStorage::Compact { data } => {
                self.collect_element_collections(
                    data,
                    dataspace.num_elements(),
                    &slots,
                    referenced,
                )?;
                true
            }
            RawReadStorage::Contiguous { address } => {
                self.collect_contiguous_collections(
                    address,
                    spec.byte_len_u64(),
                    &slots,
                    referenced,
                )?;
                true
            }
            RawReadStorage::ContiguousUnallocated => true,
            RawReadStorage::Chunked { .. } => {
                self.collect_chunked_collections(spec, &slots, referenced)?
            }
            RawReadStorage::Virtual => false,
        })
    }

    fn collect_fill_value_collections(
        &self,
        messages: &[HeaderMessage],
        slots: &ReferenceSlots,
        referenced: &mut BTreeSet<u64>,
    ) -> Result<(), FormatError> {
        for HeaderMessage {
            msg_type,
            flags,
            data,
            ..
        } in messages
        {
            if flags.is_shared()
                || !matches!(
                    *msg_type,
                    MessageType::FILL_VALUE_OLD | MessageType::FILL_VALUE
                )
            {
                continue;
            }
            if let Some(value) = FillValueView::parse(*msg_type, data)?.value() {
                self.collect_element_collections(value, 1, slots, referenced)?;
            }
        }
        Ok(())
    }

    fn collect_contiguous_collections(
        &self,
        address: StoredAddress,
        len: u64,
        slots: &ReferenceSlots,
        referenced: &mut BTreeSet<u64>,
    ) -> Result<(), FormatError> {
        let start = self.addresses.absolute(address)?;
        let element_size = NonZeroU64::from(slots.element_size);
        let batch_len = element_size.saturating_mul(ELEMENT_BATCH);
        for first in (0..len).step_by(batch_len.get().to_usize()?) {
            let offset = start
                .checked_add(first)
                .ok_or(FormatError::OffsetOverflow {
                    offset: start,
                    length: first,
                })?;
            let read_len = (len - first).min(batch_len.get());
            let bytes = self.image().read_exact_at(offset, read_len.to_usize()?)?;
            self.collect_element_collections(&bytes, read_len / element_size, slots, referenced)?;
        }
        Ok(())
    }

    /// Adds the collections that the elements of a chunked dataset refer to.
    ///
    /// The analysis reads every element of each allocated chunk, including the elements of an
    /// edge chunk past the dataset's extent. Returns `false` if a chunk cannot be read, or cannot
    /// be decoded to the size of a chunk.
    ///
    /// # Errors
    ///
    /// Returns a [`FormatError`] if the layout, the chunk index, or an element is malformed.
    fn collect_chunked_collections(
        &self,
        spec: RawReadSpec<'_>,
        slots: &ReferenceSlots,
        referenced: &mut BTreeSet<u64>,
    ) -> Result<bool, FormatError> {
        let element_size = u64::from(slots.element_size.get());
        let mut complete = true;
        crate::chunked_read::visit_decoded_chunks_from_source(
            &BaseOffsetSource {
                inner: &self.image(),
                base: self.addresses.effective_base(),
            },
            spec,
            self.widths(),
            |chunk| {
                let Ok(bytes) = chunk else {
                    complete = false;
                    return Ok(());
                };
                self.collect_element_collections(
                    bytes,
                    bytes.len() as u64 / element_size,
                    slots,
                    referenced,
                )
            },
        )?;
        Ok(complete)
    }

    /// Adds the collections that the attributes in an object's dense attribute storage refer to.
    ///
    /// Returns `false` if a record in the storage is shared, or if
    /// [`collect_attribute_collections`](Self::collect_attribute_collections) returns `false` for
    /// an attribute.
    ///
    /// # Errors
    ///
    /// Returns a [`FormatError`] if the attribute info message, the storage, or an attribute is
    /// malformed.
    fn collect_dense_attribute_collections(
        &self,
        data: &[u8],
        referenced: &mut BTreeSet<u64>,
    ) -> Result<bool, FormatError> {
        let info = AttributeInfoMessage::parse_with_offset_width(data, self.widths().offsets)?;
        let Some(heap) = info.fractal_heap_address else {
            return Ok(true);
        };
        let mut shared_records = false;
        let mut complete = true;
        crate::attribute::visit_dense_attribute_messages_from_source(
            &BaseOffsetSource {
                inner: &self.image(),
                base: self.addresses.effective_base(),
            },
            AccessMode::ReadWrite,
            self.addresses,
            &info,
            heap,
            self.widths(),
            |flags| {
                shared_records |= flags.is_shared();
                !flags.is_shared()
            },
            |data, _creation_index| {
                complete &= self.collect_attribute_collections(data, referenced)?;
                Ok(())
            },
        )?;
        Ok(complete && !shared_records)
    }

    /// Adds the collections that an attribute's elements refer to.
    ///
    /// Returns `false` if the attribute has a shared datatype or dataspace, or if
    /// [`HeapReferences::of`] returns [`HeapReferences::Unsupported`] for its datatype.
    ///
    /// # Errors
    ///
    /// Returns a [`FormatError`] if the attribute message or one of its variable-length elements
    /// is malformed.
    fn collect_attribute_collections(
        &self,
        data: &[u8],
        referenced: &mut BTreeSet<u64>,
    ) -> Result<bool, FormatError> {
        let attribute = match AttributeMessage::parse_with_length_width(data, self.widths().lengths)
        {
            Ok(attribute) => attribute,
            Err(FormatError::UnresolvedSharedMessage(_)) => return Ok(false),
            Err(error) => return Err(error),
        };
        Ok(match HeapReferences::of(&attribute.datatype)? {
            HeapReferences::Absent => true,
            HeapReferences::Located(slots) => {
                self.collect_element_collections(
                    &attribute.raw_data,
                    attribute.dataspace.num_elements(),
                    &slots,
                    referenced,
                )?;
                true
            }
            HeapReferences::Unsupported => false,
        })
    }

    /// Adds the collections that the variable-length references at `slots` in the first `count`
    /// elements of `raw` refer to.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if `count` exceeds [`usize::MAX`],
    /// [`FormatError::OffsetOverflow`] if `count` elements exceed [`usize::MAX`] bytes,
    /// [`FormatError::UnexpectedEof`] if `raw` holds fewer than `count` elements, and the errors
    /// of [`VlElement::heap_id`] for a malformed reference.
    fn collect_element_collections(
        &self,
        raw: &[u8],
        count: u64,
        slots: &ReferenceSlots,
        referenced: &mut BTreeSet<u64>,
    ) -> Result<(), FormatError> {
        let ReferenceSlots {
            element_size,
            vlen_slots,
        } = slots;
        let element_size = NonZeroU64::from(*element_size);
        let stride = element_size.get().to_usize()?;
        let len = count
            .to_usize()?
            .checked_mul(stride)
            .ok_or(FormatError::OffsetOverflow {
                offset: count,
                length: element_size.get(),
            })?;
        let elements = raw.get(..len).ok_or(FormatError::UnexpectedEof {
            expected: len,
            available: raw.len(),
        })?;
        for element in elements.chunks_exact(stride) {
            for slot in vlen_slots {
                let reference =
                    element
                        .get(slot.byte_offset..)
                        .ok_or(FormatError::UnexpectedEof {
                            expected: slot.byte_offset,
                            available: element.len(),
                        })?;
                if let Some(id) = VlElement::parse(reference, self.widths().offsets)?
                    .heap_id(self.widths().offsets, self.addresses)?
                {
                    referenced.insert(id.collection.get());
                }
            }
        }
        Ok(())
    }
}

/// How an element of a datatype refers to global heap objects.
enum HeapReferences {
    /// An element is free of file addresses.
    Absent,
    /// Every file address in an element is in one of the variable-length references of
    /// [`ReferenceSlots`].
    Located(ReferenceSlots),
    /// An element holds a file address the analysis does not locate, such as an object reference
    /// or a sequence of variable-length strings.
    Unsupported,
}

impl HeapReferences {
    /// Returns how an element of `datatype` refers to global heap objects.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] if the size of `datatype` is 0, it holds a file
    /// address, and the analysis locates each one.
    fn of(datatype: &Datatype) -> Result<Self, FormatError> {
        if !crate::datatype::datatype_holds_file_address(datatype) {
            return Ok(Self::Absent);
        }
        if holds_unlocated_address(datatype) {
            return Ok(Self::Unsupported);
        }
        let element_size = datatype.element_size()?;
        Ok(match vl_data::embedded_vlen_slots(datatype) {
            Some(vlen_slots) => Self::Located(ReferenceSlots {
                element_size,
                vlen_slots,
            }),
            None => Self::Unsupported,
        })
    }
}

/// The positions of the variable-length references in an element of a datatype.
struct ReferenceSlots {
    /// The size in bytes of an element.
    element_size: NonZeroU32,
    vlen_slots: Vec<EmbeddedVlSlot>,
}

/// Returns whether `datatype` holds a file address the analysis does not locate: a reference, or
/// a variable-length type whose base type holds a file address.
fn holds_unlocated_address(datatype: &Datatype) -> bool {
    match datatype {
        Datatype::VariableLength { base_type, .. } => {
            crate::datatype::datatype_holds_file_address(base_type)
        }
        Datatype::Compound { members, .. } => members
            .iter()
            .any(|member| holds_unlocated_address(&member.datatype)),
        Datatype::Array { base_type, .. } | Datatype::Enumeration { base_type, .. } => {
            holds_unlocated_address(base_type)
        }
        _ => crate::datatype::datatype_holds_file_address(datatype),
    }
}

/// The number of elements the analysis reads from a contiguous dataset at a time.
// The literal is non-zero, and the `unwrap` in a constant is evaluated at compile time.
const ELEMENT_BATCH: NonZeroU64 = NonZeroU64::new(4096).unwrap();
