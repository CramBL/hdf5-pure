use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::ops::Range;

use crate::__private as format_boundary;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::attribute_info::AttributeInfoMessage;
use crate::attribute_message::AttributeMessage;
use crate::bytes;
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::width::FormatWidths;

use super::attributes::{DenseAttrCreationOrder, EncodedDenseAttrs};
use super::plan::{
    DENSE_ATTR_LENGTH_WIDTH, DENSE_ATTR_OFFSET_WIDTH, DenseAttrBlockKind, DenseAttrPlan,
};

/// Dense attribute geometry and deferred payload updates.
///
/// Each encoded attribute has a creation index. Names and tracked creation indexes are
/// unique, and tracked indexes are below the retained next-index counter. The managed heap,
/// huge-object placement, index ordering, and total length are retained during
/// [`finalize`](Self::finalize).
///
/// Cloning copies encoded messages and deferred updates while sharing immutable geometry.
#[derive(Clone)]
pub struct PreparedDenseAttrs {
    plan: DenseAttrPlan,
    fixups: Vec<PreparedAttrFixup>,
}

impl PreparedDenseAttrs {
    /// Encodes the attribute set at eight-byte address and length widths.
    ///
    /// Attribute numbers in `fixups` refer to positions in `attrs`. A committed-datatype
    /// fixup requires a committed datatype reference. A global-heap fixup requires a
    /// variable-length datatype and whole references. A payload fixup requires a datatype
    /// without embedded file addresses.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if declared updates overlap, refer to an absent
    /// attribute, or do not match its representation. Propagates errors from
    /// [`dense_attrs_plan`](super::dense_attrs_plan).
    pub fn new(
        attrs: &[AttributeMessage],
        creation: DenseAttrCreationOrder,
        fixups: impl IntoIterator<Item = DenseAttrFixup>,
    ) -> Result<Self, FormatError> {
        Self::with_widths(
            FormatWidths {
                offsets: DENSE_ATTR_OFFSET_WIDTH,
                lengths: DENSE_ATTR_LENGTH_WIDTH,
            },
            attrs,
            creation,
            fixups,
        )
    }

    /// Encodes version 3 attribute bodies and prepares deferred updates at independent file widths.
    ///
    /// Variable-length raw values must already use `widths.offsets`-byte heap addresses. Fixup
    /// positions and requirements are the same as for [`Self::new`]. The managed-object limit
    /// is bounded by the direct-block capacity at these widths.
    ///
    /// # Errors
    ///
    /// Returns the metadata, geometry, and deferred-update errors of [`Self::new`], or
    /// [`FormatError::LengthTooLarge`] if a huge-object total or count exceeds `widths.lengths`.
    /// Propagates errors from [`AttributeMessage::prepare_v3`].
    pub fn with_widths(
        widths: FormatWidths,
        attrs: &[AttributeMessage],
        creation: DenseAttrCreationOrder,
        fixups: impl IntoIterator<Item = DenseAttrFixup>,
    ) -> Result<Self, FormatError> {
        Self::with_plan(
            attrs,
            DenseAttrPlan::new(EncodedDenseAttrs::new(widths, attrs, creation)?)?,
            fixups,
        )
    }

    /// Prepares deferred updates within the encoded plan.
    ///
    /// The caller supplies the same messages in `attrs` that were encoded in `plan`, in the
    /// same order. Attribute numbers in `fixups` refer to that order.
    ///
    /// # Errors
    ///
    /// Returns the deferred-update errors documented by [`Self::new`].
    fn with_plan(
        attrs: &[AttributeMessage],
        plan: DenseAttrPlan,
        fixups: impl IntoIterator<Item = DenseAttrFixup>,
    ) -> Result<Self, FormatError> {
        let widths = plan.geometry().widths();
        let reference_size = DENSE_ATTR_VL_ADDRESS_OFFSET
            + usize::from(widths.offsets.get())
            + core::mem::size_of::<u32>();
        let mut prepared = Vec::new();
        let mut seen = BTreeSet::new();
        for fixup in fixups {
            let (attribute, kind) = match fixup {
                DenseAttrFixup::CommittedDatatype { attribute } => {
                    (attribute, FixupTarget::Datatype)
                }
                DenseAttrFixup::GlobalHeap { attribute }
                | DenseAttrFixup::Payload { attribute } => (attribute, FixupTarget::Data),
            };
            let attr = attrs.get(attribute).ok_or_else(|| {
                FormatError::Internal(
                    "a dense attribute fixup refers to an absent attribute".into(),
                )
            })?;
            if !seen.insert((attribute, kind)) {
                return Err(FormatError::Internal(
                    "dense attribute fixups overlap".into(),
                ));
            }
            let bytes = plan.attributes()[attribute].bytes();
            let data_offset = bytes.len() - attr.raw_data.len();
            prepared.push(match fixup {
                DenseAttrFixup::CommittedDatatype { .. } => {
                    if !attr.datatype_location.is_committed() {
                        return Err(FormatError::Internal(
                            "a committed datatype fixup requires a committed attribute datatype".into(),
                        ));
                    }
                    let dataspace_size = usize::from(u16::from_le_bytes([
                        bytes[DENSE_ATTR_DATASPACE_SIZE_OFFSET],
                        bytes[DENSE_ATTR_DATASPACE_SIZE_OFFSET + 1],
                    ]));
                    let end = data_offset - dataspace_size;
                    PreparedAttrFixup::CommittedDatatype(FixupRange {
                        attribute,
                        bytes: end - usize::from(widths.offsets.get())..end,
                    })
                }
                DenseAttrFixup::GlobalHeap { .. } => {
                    if !matches!(attr.datatype, Datatype::VariableLength(_))
                        || attr.raw_data.len() % reference_size != 0 {
                        return Err(FormatError::Internal(
                            "a global heap fixup requires whole variable-length attribute references".into(),
                        ));
                    }
                    PreparedAttrFixup::GlobalHeap(FixupRange { attribute, bytes: data_offset..bytes.len() })
                }
                DenseAttrFixup::Payload { .. } => {
                    if format_boundary::datatype_holds_file_address(&attr.datatype) {
                        return Err(FormatError::Internal(
                            "a dense attribute payload fixup cannot replace file references".into(),
                        ));
                    }
                    PreparedAttrFixup::Payload(FixupRange { attribute, bytes: data_offset..bytes.len() })
                },
            });
        }
        Ok(Self {
            plan,
            fixups: prepared,
        })
    }

    /// Resolves the deferred updates within the retained geometry.
    ///
    /// Calls `address` with the attribute position for each committed datatype, and `payload` with
    /// the attribute position for each global-heap and whole-payload fixup. Replacement payloads
    /// must have the prepared length, and the references of a global-heap fixup must hold defined
    /// addresses. The returned [`DenseAttrPlan`] contains the resolved bytes and the original heap
    /// and index layouts.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DefinedAddressIsUndefined`] if a resolved address is undefined,
    /// [`FormatError::AddressTooLarge`] if it exceeds the retained address width, or
    /// [`FormatError::Internal`] if a replacement payload has a different length. Propagates
    /// errors reading a reference address and errors from either resolver.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_format::{CharacterSet, Datatype, FormatError, StringPadding};
    /// use hdf5_pure_format::__private::{
    ///     AttributeMessage, Dataspace, DataspaceType, DatatypeLocation,
    ///     DenseAttrCreationOrder, DenseAttrFixup, PreparedDenseAttrs,
    /// };
    ///
    /// let attribute = AttributeMessage {
    ///     name: "label".into(),
    ///     datatype: Datatype::String {
    ///         size: 1,
    ///         padding: StringPadding::NullPad,
    ///         charset: CharacterSet::Ascii,
    ///     },
    ///     dataspace: Dataspace {
    ///         space_type: DataspaceType::Scalar,
    ///         rank: 0,
    ///         dimensions: vec![],
    ///         max_dimensions: None,
    ///     },
    ///     raw_data: b"a".to_vec(),
    ///     datatype_location: DatatypeLocation::Inline,
    /// };
    /// let prepared = PreparedDenseAttrs::new(
    ///     core::slice::from_ref(&attribute),
    ///     DenseAttrCreationOrder::Untracked,
    ///     [DenseAttrFixup::Payload { attribute: 0 }],
    /// )?;
    /// assert_eq!(attribute.name, "label");
    /// assert_eq!(attribute.raw_data, b"a");
    /// let reserved = prepared.blob_len();
    /// let ready = prepared.finalize(
    ///     |attribute| panic!("unexpected committed datatype of attribute {attribute}"),
    ///     |attribute| {
    ///         assert_eq!(attribute, 0);
    ///         Ok(b"b")
    ///     },
    /// )?;
    /// let emitted = ready.build(StoredAddress::new(0x8000))?;
    /// assert_eq!(emitted.blob.len() as u64, reserved);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn finalize<'a>(
        self,
        mut address: impl FnMut(usize) -> Result<StoredAddress, FormatError>,
        mut payload: impl FnMut(usize) -> Result<&'a [u8], FormatError>,
    ) -> Result<DenseAttrPlan, FormatError> {
        let Self { mut plan, fixups } = self;
        let widths = plan.geometry().widths();
        let reference_size = DENSE_ATTR_VL_ADDRESS_OFFSET
            + usize::from(widths.offsets.get())
            + core::mem::size_of::<u32>();
        for fixup in fixups {
            match fixup {
                PreparedAttrFixup::CommittedDatatype(ref field) => {
                    let resolved = address(field.attribute)?;
                    defined_dense_attr_reference(widths, resolved)?;
                    plan.fixup_bytes_mut(field)?.copy_from_slice(
                        &resolved.get().to_le_bytes()[..usize::from(widths.offsets.get())],
                    );
                }
                PreparedAttrFixup::GlobalHeap(ref field)
                | PreparedAttrFixup::Payload(ref field) => {
                    let resolved = payload(field.attribute)?;
                    let target = plan.fixup_bytes_mut(field)?;
                    if target.len() != resolved.len() {
                        return Err(FormatError::Internal(
                            "a dense attribute fixup changes its prepared length".into(),
                        ));
                    }
                    if matches!(fixup, PreparedAttrFixup::GlobalHeap(_)) {
                        for reference in resolved.chunks_exact(reference_size) {
                            defined_dense_attr_reference(
                                widths,
                                StoredAddress::new(bytes::read_offset_width(
                                    reference,
                                    DENSE_ATTR_VL_ADDRESS_OFFSET,
                                    widths.offsets,
                                )?),
                            )?;
                        }
                    }
                    target.copy_from_slice(resolved);
                }
            }
        }
        Ok(plan)
    }

    /// Returns the allocation kind of each block in placement order.
    ///
    /// The order matches [`Self::block_lengths`] and is preserved by [`Self::finalize`].
    pub fn block_kinds(&self) -> impl ExactSizeIterator<Item = DenseAttrBlockKind> + '_ {
        self.plan.block_kinds()
    }

    /// Returns the complete block lengths before resolving deferred updates.
    ///
    /// The lengths and their order match [`DenseAttrPlan::block_lengths`] after
    /// [`Self::finalize`].
    pub fn block_lengths(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.plan.block_lengths()
    }

    /// Returns the object-header metadata for independently placed blocks.
    ///
    /// Pass one address per entry of [`Self::block_lengths`], in the same order. The returned
    /// message contains the placement addresses and creation metadata, independently of the
    /// deferred payload updates. Resolve those updates with [`Self::finalize`] before
    /// serializing the blocks through
    /// [`DenseAttrPlacement::serialize_blocks`](super::DenseAttrPlacement::serialize_blocks).
    ///
    /// # Errors
    ///
    /// Returns the placement errors documented by [`DenseAttrPlan::place`].
    pub fn attribute_info(
        &self,
        addresses: &[StoredAddress],
    ) -> Result<AttributeInfoMessage, FormatError> {
        self.plan.placement_info(addresses)
    }

    /// Returns the widths retained for reference resolution, placement, and serialization.
    pub fn widths(&self) -> FormatWidths {
        self.plan.geometry().widths()
    }

    /// Returns the reserved length, including heap blocks, indexes, and huge objects.
    pub fn blob_len(&self) -> u64 {
        self.plan.blob_len()
    }

    /// Returns the object-header metadata for a heap placed at `heap_address`.
    ///
    /// # Errors
    ///
    /// Returns the address errors documented by [`DenseAttrPlan::attribute_info_for_heap`].
    pub fn attribute_info_for_heap(
        &self,
        heap_address: StoredAddress,
    ) -> Result<AttributeInfoMessage, FormatError> {
        self.plan.attribute_info_for_heap(heap_address)
    }
}

/// A deferred change to one attribute's encoded bytes.
///
/// Attribute positions refer to the input order of [`PreparedDenseAttrs::new`] and
/// [`PreparedDenseAttrs::with_widths`].
#[derive(Clone, Copy, Debug)]
pub enum DenseAttrFixup {
    /// Replaces the committed datatype's file address.
    CommittedDatatype {
        /// The attribute position in the prepared set.
        attribute: usize,
    },
    /// Replaces variable-length references, collection addresses and object indexes, with the
    /// attribute's resolved raw data.
    GlobalHeap {
        /// The attribute position in the prepared set.
        attribute: usize,
    },
    /// Replaces raw data with a payload of the same length.
    Payload {
        /// The attribute position in the prepared set.
        attribute: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum FixupTarget {
    Data,
    Datatype,
}

/// A declared fixup with its offset in the encoded attribute message.
#[derive(Clone, Debug)]
enum PreparedAttrFixup {
    CommittedDatatype(FixupRange),
    GlobalHeap(FixupRange),
    Payload(FixupRange),
}

/// The input position of an attribute and the range to replace within its encoded message.
#[derive(Clone, Debug)]
pub(super) struct FixupRange {
    attribute: usize,
    bytes: Range<usize>,
}

impl FixupRange {
    pub(super) fn attribute(&self) -> usize {
        self.attribute
    }
    pub(super) fn bytes(&self) -> Range<usize> {
        self.bytes.clone()
    }
}

/// Checks a resolved reference against the retained address width.
///
/// # Errors
///
/// Returns [`FormatError::DefinedAddressIsUndefined`] if `address` is undefined, or
/// [`FormatError::AddressTooLarge`] if it exceeds `widths.offsets`.
fn defined_dense_attr_reference(
    widths: FormatWidths,
    address: StoredAddress,
) -> Result<(), FormatError> {
    if address.get() > StoredAddress::undefined(widths.offsets).get() {
        return Err(FormatError::AddressTooLarge {
            address: address.get(),
            offset_size: widths.offsets.get(),
        });
    }
    if address.is_undefined(widths.offsets) {
        return Err(FormatError::DefinedAddressIsUndefined {
            address: address.get(),
            offset_size: widths.offsets.get(),
        });
    }
    Ok(())
}

const DENSE_ATTR_VL_ADDRESS_OFFSET: usize = 4;

// The dataspace-size field begins at byte six of a version 3 Attribute message
// ("The Attribute Message", format specification, version 4.0).
const DENSE_ATTR_DATASPACE_SIZE_OFFSET: usize = 6;

#[cfg(test)]
#[path = "tests/operations.rs"]
mod tests;
