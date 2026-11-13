//! Variable-length strings and sequences stored in the global heap.
//!
//! Each stored descriptor holds a base-element count and a global heap ID. [`VlElement::parse`]
//! reads its fields through [`VariableLengthDescriptor`], and [`VlElement::heap_id`] resolves
//! its member identity.

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};

use core::num::NonZeroU16;

use hdf5_pure_core::__private::ElementCount;
use hdf5_pure_core::__private::ElementSize;
use hdf5_pure_format::__private::VariableLengthDescriptor;

#[cfg(test)]
use crate::FixedPointType;
use crate::VariableLengthType;
use crate::address::AddressContext;
#[cfg(test)]
use crate::address::BaseAddress;
#[cfg(test)]
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
#[cfg(test)]
use crate::address::StoredAddressExt;
use crate::convert::Narrow;
use crate::datatype::{CharacterSet, Datatype};
use crate::error::FormatError;
use crate::global_heap::GlobalHeapId;
use crate::global_heap::GlobalHeapReader;
#[cfg(test)]
use crate::source::BytesSource;
use crate::source::Source;
use crate::width::{FormatWidths, OffsetWidth};

/// Allocation limits for reading variable-length strings.
///
/// Limits are checked before any string payload is materialized. The payload
/// byte limit counts referenced string bytes. Allocation metadata for the
/// vector and strings is excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VlenStringReadOptions {
    max_elements: Option<usize>,
    max_payload_bytes: Option<usize>,
}

impl VlenStringReadOptions {
    /// Creates options with no limits.
    pub const fn new() -> Self {
        Self {
            max_elements: None,
            max_payload_bytes: None,
        }
    }

    /// Sets the maximum number of VL elements that may be read.
    pub const fn with_max_elements(mut self, max_elements: usize) -> Self {
        self.max_elements = Some(max_elements);
        self
    }

    /// Sets the maximum total string payload size in bytes.
    pub const fn with_max_payload_bytes(mut self, max_payload_bytes: usize) -> Self {
        self.max_payload_bytes = Some(max_payload_bytes);
        self
    }

    /// Returns the configured element limit.
    pub const fn max_elements(&self) -> Option<usize> {
        self.max_elements
    }

    /// Returns the configured payload-byte limit.
    pub const fn max_payload_bytes(&self) -> Option<usize> {
        self.max_payload_bytes
    }
}

/// A parsed base-element count and stored global heap ID.
#[derive(Debug, Clone)]
pub struct VlElement {
    /// The number of base elements, independent of their payload byte length.
    pub length: u32,
    /// The stored base-relative collection address, including zero and undefined values.
    pub collection_address: StoredAddress,
    /// The full four-byte object index before conversion to a nonzero member index.
    pub object_index: u32,
}

impl VlElement {
    /// Returns eight bytes plus the file's address width.
    pub(crate) fn serialized_size(offset_width: OffsetWidth) -> usize {
        VariableLengthDescriptor::serialized_size(offset_width)
    }

    /// Parses one complete stored descriptor from the start of `raw`.
    ///
    /// Retains the base-element count, stored collection address, and full object index through
    /// [`VariableLengthDescriptor`]. Trailing bytes remain unexamined. The caller uses
    /// [`Self::heap_id`] to interpret null references and resolve a member identity.
    ///
    /// The global heap ID is defined in "Global Heap" of the [format specification, version
    /// 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `raw` is shorter than `offset_width.get() + 8`
    /// bytes.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
    pub(crate) fn parse(raw: &[u8], offset_width: OffsetWidth) -> Result<Self, FormatError> {
        let (descriptor, _) = VariableLengthDescriptor::parse(offset_width, raw)?;
        Ok(Self {
            length: descriptor.element_count(),
            collection_address: descriptor.collection_address(),
            object_index: descriptor.object_index(),
        })
    }

    /// Returns the ID of the global heap object the element refers to, or `None` for a null
    /// element, one of length 0 whose collection address is undefined or 0.
    ///
    /// Resolves a defined, nonzero collection address even when `length` is zero.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::VlDataError`] if the element is not null and its collection address
    /// is undefined or its object index does not fit a `u16`,
    /// [`FormatError::OffsetOverflow`] if the collection address resolves past [`u64::MAX`], and
    /// [`FormatError::GlobalHeapObjectNotFound`] if the object index is 0.
    pub(crate) fn heap_id(
        &self,
        offset_width: OffsetWidth,
        addresses: AddressContext,
    ) -> Result<Option<GlobalHeapId>, FormatError> {
        let undefined = self.collection_address.is_undefined(offset_width);
        if self.length == 0 && (undefined || self.collection_address.get() == 0) {
            return Ok(None);
        }
        if undefined {
            return Err(FormatError::VlDataError(
                "non-empty VL element has an undefined heap address".into(),
            ));
        }
        let collection = addresses.resolve(self.collection_address)?;
        let index: u16 = self.object_index.narrow_or_else(|| {
            FormatError::VlDataError(format!(
                "global heap object index {} does not fit u16",
                self.object_index
            ))
        })?;
        let index = NonZeroU16::new(index).ok_or(FormatError::GlobalHeapObjectNotFound {
            collection_address: collection.get(),
            index,
        })?;
        Ok(Some(GlobalHeapId { collection, index }))
    }

    /// Returns the size in bytes of the element's data, `length * element_size`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the size exceeds [`u64::MAX`].
    fn byte_len(&self, element_size: usize) -> Result<u64, FormatError> {
        u64::from(self.length)
            .checked_mul(element_size as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset: u64::from(self.length),
                length: element_size as u64,
            })
    }
}

/// Parses the first `num_elements` variable-length references in `raw_data`.
///
/// Each reference uses `offset_width` for its collection address. Trailing bytes
/// after those `num_elements` references are ignored.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if the encoded extent exceeds [`u64::MAX`],
/// [`FormatError::ValueTooLargeForPlatform`] if a count or extent exceeds [`usize::MAX`],
/// or [`FormatError::UnexpectedEof`] if `raw_data` is shorter than the required extent.
pub fn parse_vl_references(
    raw_data: &[u8],
    num_elements: ElementCount,
    offset_width: OffsetWidth,
) -> Result<Vec<VlElement>, FormatError> {
    let elem_size = ElementSize::new(VlElement::serialized_size(offset_width) as u64)
        .ok_or_else(|| FormatError::Internal("VL reference element size is zero".into()))?;
    let total = elem_size
        .byte_length(num_elements)
        .ok_or(FormatError::OffsetOverflow {
            offset: num_elements.get(),
            length: elem_size.get(),
        })?
        .to_usize()?;
    if raw_data.len() < total {
        return Err(FormatError::UnexpectedEof {
            expected: total,
            available: raw_data.len(),
        });
    }

    let mut elements = Vec::with_capacity(num_elements.to_usize()?);
    for raw in raw_data[..total].chunks_exact(elem_size.get().to_usize()?) {
        elements.push(VlElement::parse(raw, offset_width)?);
    }
    Ok(elements)
}

/// Returns `true` for VL strings and sequences of one-byte ASCII strings.
pub(crate) fn is_vlen_string_datatype(datatype: &Datatype) -> bool {
    match datatype {
        Datatype::VariableLength(VariableLengthType::String(_)) => true,
        Datatype::VariableLength(VariableLengthType::Sequence { element_type }) => matches!(
            element_type.as_ref(),
            Datatype::String {
                size: 1,
                charset: CharacterSet::Ascii,
                ..
            }
        ),
        _ => false,
    }
}

/// A variable-length reference within a compound member or array entry.
///
/// Readers use its position and element size to inspect heap references. The repack writer uses
/// them to copy heap payloads and rewrite references in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EmbeddedVlSlot {
    /// Byte offset of the reference within one stored element.
    pub byte_offset: usize,
    /// The byte multiplier for the reference's stored `length`. This is 1 for strings
    /// and the base-type element size for sequences.
    pub element_size: usize,
}

/// Dataset bytes and heap payloads for rewriting embedded variable-length references.
pub(crate) struct EmbeddedVlData {
    /// The dataset's element bytes, references still carrying source addresses.
    pub raw: Vec<u8>,
    /// Byte offset within `raw` of each embedded reference.
    pub offsets: Vec<usize>,
    /// The heap payload for each reference, paired in order with `offsets`.
    pub objects: Vec<VlByteObject>,
}

/// Returns reference slots at eight-byte address width in compound and array declaration order.
///
/// Returns `None` if a stored size exceeds its field or platform capacity, or if slot traversal
/// exceeds address or storage bounds. A top-level variable-length datatype has one slot at
/// offset 0.
pub(crate) fn embedded_vlen_slots(datatype: &Datatype) -> Option<Vec<EmbeddedVlSlot>> {
    embedded_vlen_slots_with_offset_width(datatype, OffsetWidth::Eight)
        .ok()
        .flatten()
}

/// Returns bounded reference slots using stored sizes at `offset_width`.
///
/// Walks compound members and array entries in declaration order. A top-level variable-length
/// value has one slot at offset 0. Returns `Ok(None)` if address arithmetic, slot capacity, or
/// bounds for complete references cannot be established.
///
/// # Errors
///
/// Returns the field-size errors of [`crate::datatype::stored_type_size`], or
/// [`FormatError::ValueTooLargeForPlatform`] if stored geometry exceeds [`usize::MAX`].
pub(crate) fn embedded_vlen_slots_with_offset_width(
    datatype: &Datatype,
    offset_width: OffsetWidth,
) -> Result<Option<Vec<EmbeddedVlSlot>>, FormatError> {
    let element_size = crate::datatype::stored_type_size(datatype, offset_width)?.to_usize()?;
    let reference_size = VlElement::serialized_size(offset_width);
    let capacity = element_size / reference_size;
    let mut slots = Vec::new();
    if !collect_vlen_slots(datatype, offset_width, 0, capacity, &mut slots)? {
        return Ok(None);
    }
    if slots.len() > capacity
        || slots.iter().any(|s| {
            s.byte_offset
                .checked_add(reference_size)
                .is_none_or(|end| end > element_size)
        })
    {
        return Ok(None);
    }
    Ok(Some(slots))
}

/// Reference stride for test fixtures with eight-byte heap addresses.
#[cfg(test)]
const VL_REF_BYTES: usize = 16;

/// Appends reference slots while checking address arithmetic and output capacity.
///
/// Returns `Ok(false)` if address arithmetic fails or an array with reference slots has more
/// entries than `capacity`. Appends at most one excess slot for the caller's capacity check.
/// Previously appended slots remain in `out` on error.
///
/// # Errors
///
/// Returns the field-size errors of [`crate::datatype::stored_type_size`], or
/// [`FormatError::ValueTooLargeForPlatform`] if a stored size exceeds [`usize::MAX`].
fn collect_vlen_slots(
    datatype: &Datatype,
    offset_width: OffsetWidth,
    base: usize,
    capacity: usize,
    out: &mut Vec<EmbeddedVlSlot>,
) -> Result<bool, FormatError> {
    // Stop after one excess slot so the caller detects capacity overflow.
    if out.len() > capacity {
        return Ok(true);
    }
    Ok(match datatype {
        Datatype::VariableLength(kind) => {
            let base_type = kind.base_type();
            // A VL string's reference counts bytes. A sequence's reference counts
            // base-type elements.
            let element_size = if is_vlen_string_datatype(datatype) {
                1
            } else {
                crate::datatype::stored_type_size(base_type, offset_width)?
                    .to_usize()?
                    .max(1)
            };
            out.push(EmbeddedVlSlot {
                byte_offset: base,
                element_size,
            });
            true
        }
        Datatype::Compound { members, .. } => {
            for m in members {
                let Some(at) = usize::try_from(m.byte_offset)
                    .ok()
                    .and_then(|off| base.checked_add(off))
                else {
                    return Ok(false);
                };
                if !collect_vlen_slots(&m.datatype, offset_width, at, capacity, out)? {
                    return Ok(false);
                }
            }
            true
        }
        Datatype::Array {
            base_type,
            dimensions,
        } => {
            // Walk the base type once and translate its slots to each array entry.
            // Each repeated entry contributes a slot, so the output bound also limits the work.
            let mut probe = Vec::new();
            if !collect_vlen_slots(base_type, offset_width, 0, capacity, &mut probe)? {
                return Ok(false);
            }
            // No reference in one entry means none in any repetition of it.
            if probe.is_empty() {
                return Ok(true);
            }
            let count = dimensions
                .iter()
                .copied()
                .fold(1u64, |a, b| a.saturating_mul(u64::from(b)));
            // Reject an entry count larger than the slot capacity before allocation.
            if count > capacity as u64 {
                return Ok(false);
            }
            let entries = usize::try_from(count).unwrap_or(usize::MAX);
            let stride = crate::datatype::stored_type_size(base_type, offset_width)?.to_usize()?;
            for i in 0..entries {
                let Some(at) = i.checked_mul(stride).and_then(|off| base.checked_add(off)) else {
                    return Ok(false);
                };
                for slot in &probe {
                    let Some(byte_offset) = at.checked_add(slot.byte_offset) else {
                        return Ok(false);
                    };
                    out.push(EmbeddedVlSlot {
                        byte_offset,
                        element_size: slot.element_size,
                    });
                    // Stop after one excess slot so the caller detects capacity overflow.
                    if out.len() > capacity {
                        return Ok(true);
                    }
                }
            }
            true
        }
        // An enumeration's base is an integer, and no other class can reach a
        // variable-length reference.
        _ => true,
    })
}

fn check_element_limit(
    num_elements: ElementCount,
    options: VlenStringReadOptions,
) -> Result<(), FormatError> {
    if let Some(limit) = options.max_elements
        && num_elements > ElementCount::new(limit as u64)
    {
        return Err(FormatError::VariableLengthElementLimitExceeded {
            limit,
            actual: num_elements.get(),
        });
    }
    Ok(())
}

fn payload_size(refs: &[VlElement], options: VlenStringReadOptions) -> Result<u64, FormatError> {
    let mut required = 0u64;
    for element in refs {
        required =
            required
                .checked_add(u64::from(element.length))
                .ok_or(FormatError::OffsetOverflow {
                    offset: required,
                    length: u64::from(element.length),
                })?;
    }
    if let Some(limit) = options.max_payload_bytes
        && required > limit as u64
    {
        return Err(FormatError::VariableLengthByteLimitExceeded { limit, required });
    }
    Ok(required)
}

/// Returns the summed payload length of the first `num_elements` VL references.
///
/// # Errors
///
/// Returns the errors from [`parse_vl_references`], or
/// [`FormatError::OffsetOverflow`] if the summed payload length exceeds [`u64::MAX`].
pub fn vlen_string_payload_size(
    raw_data: &[u8],
    num_elements: ElementCount,
    offset_width: OffsetWidth,
) -> Result<u64, FormatError> {
    check_element_limit(num_elements, VlenStringReadOptions::default())?;
    let refs = parse_vl_references(raw_data, num_elements, offset_width)?;
    payload_size(&refs, VlenStringReadOptions::default())
}

/// Reads VL strings from `source` and passes each decoded string to `visitor`.
///
/// References are visited in order. Null references produce empty strings, and
/// invalid UTF-8 is replaced lossily. Element and payload limits are checked
/// before any heap payload is read.
///
/// # Errors
///
/// Returns the errors from [`checked_references`] or [`visit_vl_objects`]. The
/// visitor receives the strings preceding the first heap-read or reference error.
pub fn visit_vl_strings_from_source<S, F>(
    source: &S,
    raw_data: &[u8],
    num_elements: ElementCount,
    widths: FormatWidths,
    addresses: AddressContext,
    options: VlenStringReadOptions,
    mut visitor: F,
) -> Result<(), FormatError>
where
    S: Source + ?Sized,
    F: FnMut(&str),
{
    let refs = checked_references(raw_data, num_elements, widths.offsets, options)?;
    visit_vl_objects(source, &refs, widths, addresses, 1, |object| match object {
        VlByteObject::Null => visitor(""),
        VlByteObject::Bytes(bytes) => visitor(&String::from_utf8_lossy(&bytes)),
    })
}

/// Reads VL strings from `source` in reference order.
///
/// Null references produce empty strings, and invalid UTF-8 is replaced lossily.
///
/// # Errors
///
/// Returns the errors from [`visit_vl_strings_from_source`].
pub fn read_vl_strings_from_source<S: Source + ?Sized>(
    source: &S,
    raw_data: &[u8],
    num_elements: ElementCount,
    widths: FormatWidths,
    addresses: AddressContext,
    options: VlenStringReadOptions,
) -> Result<Vec<String>, FormatError> {
    let mut strings = Vec::new();
    visit_vl_strings_from_source(
        source,
        raw_data,
        num_elements,
        widths,
        addresses,
        options,
        |string| strings.push(String::from(string)),
    )?;
    Ok(strings)
}

/// A null reference or exact heap payload from a VL dataset or attribute.
///
/// [`Null`](Self::Null) is a zero-length reference with an undefined or zero
/// collection address. [`Bytes`](Self::Bytes) contains an actual heap object's
/// bytes, including empty payloads, embedded NULs, or invalid UTF-8.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VlByteObject {
    /// A null VL reference (no heap object).
    Null,
    /// A heap object holding these exact bytes.
    Bytes(Vec<u8>),
}

/// Reads exact heap payloads while preserving null references.
///
/// `element_size` is 1 for strings and the base-type size for sequences. Each
/// payload occupies `length * element_size` bytes, including embedded NULs or
/// invalid UTF-8.
///
/// # Errors
///
/// Returns the errors from [`checked_references`] or [`visit_vl_objects`].
pub(crate) fn read_vl_byte_objects_from_source<S: Source + ?Sized>(
    source: &S,
    raw_data: &[u8],
    num_elements: ElementCount,
    widths: FormatWidths,
    addresses: AddressContext,
    element_size: usize,
    options: VlenStringReadOptions,
) -> Result<Vec<VlByteObject>, FormatError> {
    let refs = checked_references(raw_data, num_elements, widths.offsets, options)?;
    let mut objects = Vec::with_capacity(refs.len());
    visit_vl_objects(source, &refs, widths, addresses, element_size, |object| {
        objects.push(object)
    })?;
    Ok(objects)
}

/// Parses the `num_elements` references in `raw_data` and checks them against the limits of
/// `options`.
///
/// # Errors
///
/// Returns [`FormatError::VariableLengthElementLimitExceeded`] if `num_elements` exceeds the
/// element limit, [`FormatError::VariableLengthByteLimitExceeded`] if the lengths of the elements
/// add up past the byte limit, and the errors [`parse_vl_references`] returns.
fn checked_references(
    raw_data: &[u8],
    num_elements: ElementCount,
    offset_width: OffsetWidth,
    options: VlenStringReadOptions,
) -> Result<Vec<VlElement>, FormatError> {
    check_element_limit(num_elements, options)?;
    let refs = parse_vl_references(raw_data, num_elements, offset_width)?;
    payload_size(&refs, options)?;
    Ok(refs)
}

/// Reads the object each element of `refs` refers to and passes it to `visitor`, in the order of
/// `refs`.
///
/// `visitor` receives [`VlByteObject::Null`] for a null element. Each base-type element of a
/// sequence is `element_size` bytes. The function passes the ID of every non-null element to
/// [`GlobalHeapReader::new`] before the first read, so the reader walks each collection once.
///
/// # Errors
///
/// Returns the first error of an element, in the order of `refs`: the errors
/// [`VlElement::heap_id`],
/// [`VlElement::byte_len`] and [`GlobalHeapReader::read`] return. `visitor` receives the objects of
/// the elements before the one that fails.
fn visit_vl_objects(
    source: &(impl Source + ?Sized),
    refs: &[VlElement],
    widths: FormatWidths,
    addresses: AddressContext,
    element_size: usize,
    mut visitor: impl FnMut(VlByteObject),
) -> Result<(), FormatError> {
    let mut heap = GlobalHeapReader::new(
        source,
        widths.lengths,
        refs.iter()
            .filter_map(|element| element.heap_id(widths.offsets, addresses).ok().flatten()),
    );
    for element in refs {
        visitor(match element.heap_id(widths.offsets, addresses)? {
            None => VlByteObject::Null,
            Some(id) => VlByteObject::Bytes(heap.read(id, element.byte_len(element_size)?)?),
        });
    }
    Ok(())
}

/// Reads VL strings from an in-memory file image with a zero base address.
///
/// # Errors
///
/// Returns the errors from [`read_vl_strings_from_source`].
#[cfg(test)]
pub fn read_vl_strings(
    file_data: &[u8],
    raw_data: &[u8],
    num_elements: ElementCount,
    widths: FormatWidths,
) -> Result<Vec<String>, FormatError> {
    read_vl_strings_from_source(
        &BytesSource::new(file_data),
        raw_data,
        num_elements,
        widths,
        AddressContext::from_effective_base(BaseAddress::ZERO),
        VlenStringReadOptions::default(),
    )
}

#[cfg(test)]
mod tests;
