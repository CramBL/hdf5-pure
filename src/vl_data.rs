//! Variable-length data reading (VL strings & VL sequences).
//!
//! VL data elements in HDF5 store their values in the global heap.
//! The raw data for each element contains a global heap ID:
//! `sequence_length(4 LE) + collection_address(offset_size LE) + object_index(4 LE)`.

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};

use core::num::NonZeroU16;

use crate::address::AddressContext;
#[cfg(test)]
use crate::address::BaseAddress;
#[cfg(test)]
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::bytes;
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
/// byte limit covers the bytes referenced by the VL elements; it excludes the
/// `Vec<String>` and `String` allocation metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VlenStringReadOptions {
    max_elements: Option<usize>,
    max_payload_bytes: Option<usize>,
}

impl VlenStringReadOptions {
    /// Create options with no limits.
    pub const fn new() -> Self {
        Self {
            max_elements: None,
            max_payload_bytes: None,
        }
    }

    /// Set the maximum number of VL elements that may be read.
    pub const fn with_max_elements(mut self, max_elements: usize) -> Self {
        self.max_elements = Some(max_elements);
        self
    }

    /// Set the maximum total string payload size in bytes.
    pub const fn with_max_payload_bytes(mut self, max_payload_bytes: usize) -> Self {
        self.max_payload_bytes = Some(max_payload_bytes);
        self
    }

    /// Return the configured element limit.
    pub const fn max_elements(&self) -> Option<usize> {
        self.max_elements
    }

    /// Return the configured payload-byte limit.
    pub const fn max_payload_bytes(&self) -> Option<usize> {
        self.max_payload_bytes
    }
}

/// A parsed variable-length element reference (global heap ID).
#[derive(Debug, Clone)]
pub struct VlElement {
    /// Length of the VL data.
    pub length: u32,
    /// Address of the global heap collection containing the data.
    pub collection_address: StoredAddress,
    /// Index of the object within the collection.
    pub object_index: u32,
}

impl VlElement {
    pub(crate) fn serialized_size(offset_width: OffsetWidth) -> usize {
        SEQUENCE_LENGTH_BYTES + usize::from(offset_width.get()) + HEAP_INDEX_BYTES
    }

    /// Parses the variable-length element at the start of `raw`: a 4-byte length, then the global
    /// heap ID of its data, a collection address of `offset_width.get()` bytes and a 4-byte object
    /// index.
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
        let index_at = SEQUENCE_LENGTH_BYTES + usize::from(offset_width.get());
        bytes::ensure_len(raw, 0, Self::serialized_size(offset_width))?;
        Ok(Self {
            length: u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
            collection_address: StoredAddress::new(bytes::read_offset_width(
                raw,
                SEQUENCE_LENGTH_BYTES,
                offset_width,
            )?),
            object_index: u32::from_le_bytes([
                raw[index_at],
                raw[index_at + 1],
                raw[index_at + 2],
                raw[index_at + 3],
            ]),
        })
    }

    /// Returns the ID of the global heap object the element refers to, or `None` for a null
    /// element, one of length 0 whose collection address is undefined or 0.
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

    /// Returns the size in bytes of the element's data, `length` base-type elements of
    /// `element_size` bytes each.
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

/// Parse VL global heap references from raw attribute/dataset data.
pub fn parse_vl_references(
    raw_data: &[u8],
    num_elements: u64,
    offset_width: OffsetWidth,
) -> Result<Vec<VlElement>, FormatError> {
    let elem_size = VlElement::serialized_size(offset_width) as u64;
    let total = num_elements
        .checked_mul(elem_size)
        .ok_or(FormatError::OffsetOverflow {
            offset: num_elements,
            length: elem_size,
        })?
        .to_usize()?;
    if raw_data.len() < total {
        return Err(FormatError::UnexpectedEof {
            expected: total,
            available: raw_data.len(),
        });
    }

    let mut elements = Vec::with_capacity(num_elements.to_usize()?);
    for raw in raw_data[..total].chunks_exact(elem_size.to_usize()?) {
        elements.push(VlElement::parse(raw, offset_width)?);
    }
    Ok(elements)
}

/// Whether a datatype is one of the string-shaped VL encodings understood by
/// this module.
pub(crate) fn is_vlen_string_datatype(datatype: &Datatype) -> bool {
    match datatype {
        Datatype::VariableLength {
            is_string: true, ..
        } => true,
        Datatype::VariableLength {
            is_string: false,
            base_type,
            ..
        } => matches!(
            base_type.as_ref(),
            Datatype::String {
                size: 1,
                charset: CharacterSet::Ascii,
                ..
            }
        ),
        _ => false,
    }
}

/// A variable-length reference reached *inside* a larger element rather than
/// being the element itself: a member of a compound, or an entry of an array of
/// them. Repack needs both coordinates to re-stage the payload and rewrite the
/// reference in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EmbeddedVlSlot {
    /// Byte offset of the 16-byte reference within one element of the dataset.
    pub byte_offset: usize,
    /// Byte width of the variable-length base type. The reference's stored
    /// `length` counts base-type elements, so the heap object holds
    /// `length * element_size` bytes.
    pub element_size: usize,
}

/// A dataset with embedded variable-length references, read for rewriting: the
/// element bytes as they stand, plus each reference's position within them and
/// the heap payload it names.
pub(crate) struct EmbeddedVlData {
    /// The dataset's element bytes, references still carrying source addresses.
    pub raw: Vec<u8>,
    /// Byte offset within `raw` of each embedded reference.
    pub offsets: Vec<usize>,
    /// The heap payload each reference names, paired in order with `offsets`.
    pub objects: Vec<VlByteObject>,
}

/// Every variable-length reference `datatype` reaches through a compound member
/// or array entry, in declaration order.
///
/// A datatype that *is* variable-length yields the single slot at offset 0, so
/// callers that handle the top-level case separately should test for that first.
/// Returns `None` if the datatype's declared size cannot hold the slots found,
/// which means either a malformed datatype or dimensions that overflowed — in
/// both cases the element bytes cannot be walked safely.
pub(crate) fn embedded_vlen_slots(datatype: &Datatype) -> Option<Vec<EmbeddedVlSlot>> {
    // A reference is 16 bytes, so an element can hold no more than this many.
    // Bounding the walk keeps a datatype declaring absurd array dimensions from
    // driving an unbounded allocation here.
    let element_size = datatype.type_size() as usize;
    let capacity = element_size / VL_REF_BYTES;
    let mut slots = Vec::new();
    if !collect_vlen_slots(datatype, 0, capacity, &mut slots) {
        return None;
    }
    // `checked_add`: an offset near the top of the address space would otherwise
    // wrap here and read as "fits".
    if slots.len() > capacity
        || slots.iter().any(|s| {
            s.byte_offset
                .checked_add(VL_REF_BYTES)
                .is_none_or(|end| end > element_size)
        })
    {
        return None;
    }
    Some(slots)
}

/// On-disk width of a variable-length reference with 8-byte offsets, which is
/// the only offset size the write path emits and the only one repack re-stages.
const VL_REF_BYTES: usize = 16;

// H5T__vlen_disk_write writes the sequence length as a u32 in H5Tvlen.c, HDF5 2.2.0.
const SEQUENCE_LENGTH_BYTES: usize = 4;

// The object index occupies four bytes: section `subsec_fmt4_infra_globalheap`, version 4.0.
const HEAP_INDEX_BYTES: usize = 4;

/// Walk `datatype`, appending a slot for each variable-length reference it
/// reaches. Returns `false` if it cannot be walked on this target: an offset the
/// file declares that does not fit `usize`, or one that overflows it, names a
/// position this platform cannot address, and silently truncating it would place
/// a slot somewhere plausible but wrong.
fn collect_vlen_slots(
    datatype: &Datatype,
    base: usize,
    capacity: usize,
    out: &mut Vec<EmbeddedVlSlot>,
) -> bool {
    // Stop one past the bound so the caller can still tell "too many" from "fits".
    if out.len() > capacity {
        return true;
    }
    match datatype {
        Datatype::VariableLength { base_type, .. } => {
            // A VL string's reference counts bytes (its base type is one byte
            // wide); a sequence's counts base-type elements.
            let element_size = if is_vlen_string_datatype(datatype) {
                1
            } else {
                (base_type.type_size() as usize).max(1)
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
                    return false;
                };
                if !collect_vlen_slots(&m.datatype, at, capacity, out) {
                    return false;
                }
            }
            true
        }
        Datatype::Array {
            base_type,
            dimensions,
        } => {
            // If the entry type reaches no reference then no repetition of it
            // does either. Checking once keeps a datatype declaring huge
            // dimensions from spinning through entries that can never contribute,
            // and makes every iteration below push at least one slot — so the
            // `capacity` bound terminates the loop.
            // Walk the entry type *once*, at offset 0, then translate that result
            // to each entry's base. Re-walking per entry would recompute the same
            // sub-tree `entries` times at every level of nesting, which is
            // exponential in nesting depth: a ~300-byte datatype of nested arrays
            // is enough to burn hours of CPU on work whose output is bounded by
            // `capacity`.
            let mut probe = Vec::new();
            if !collect_vlen_slots(base_type, 0, capacity, &mut probe) {
                return false;
            }
            // No reference in one entry means none in any repetition of it.
            if probe.is_empty() {
                return true;
            }
            let count = dimensions
                .iter()
                .copied()
                .fold(1u64, |a, b| a.saturating_mul(u64::from(b)));
            // Every entry contributes at least one slot, so more entries than the
            // element has room for cannot fit however the walk goes. Reject up
            // front rather than materializing that many slots for the caller to
            // discard: for a datatype whose declared size saturates, `capacity`
            // runs to hundreds of millions, which is merely wasteful on a 64-bit
            // target and an outright allocation failure on a 32-bit one.
            if count > capacity as u64 {
                return false;
            }
            let entries = usize::try_from(count).unwrap_or(usize::MAX);
            let stride = base_type.type_size() as usize;
            for i in 0..entries {
                let Some(at) = i.checked_mul(stride).and_then(|off| base.checked_add(off)) else {
                    return false;
                };
                for slot in &probe {
                    let Some(byte_offset) = at.checked_add(slot.byte_offset) else {
                        return false;
                    };
                    out.push(EmbeddedVlSlot {
                        byte_offset,
                        element_size: slot.element_size,
                    });
                    // Stop one past the bound so the caller can still tell
                    // "too many" from "fits".
                    if out.len() > capacity {
                        return true;
                    }
                }
            }
            true
        }
        // An enumeration's base is an integer, and no other class can reach a
        // variable-length reference.
        _ => true,
    }
}

fn check_element_limit(
    num_elements: u64,
    options: VlenStringReadOptions,
) -> Result<(), FormatError> {
    if let Some(limit) = options.max_elements
        && num_elements > limit as u64
    {
        return Err(FormatError::VariableLengthElementLimitExceeded {
            limit,
            actual: num_elements,
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

/// Return the total payload bytes named by a set of VL references.
pub fn vlen_string_payload_size(
    raw_data: &[u8],
    num_elements: u64,
    offset_width: OffsetWidth,
) -> Result<u64, FormatError> {
    check_element_limit(num_elements, VlenStringReadOptions::default())?;
    let refs = parse_vl_references(raw_data, num_elements, offset_width)?;
    payload_size(&refs, VlenStringReadOptions::default())
}

/// Resolve VL strings from a random-access file source and pass them to a
/// visitor one at a time.
pub fn visit_vl_strings_from_source<S, F>(
    source: &S,
    raw_data: &[u8],
    num_elements: u64,
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

/// Resolve VL strings from a random-access file source.
pub fn read_vl_strings_from_source<S: Source + ?Sized>(
    source: &S,
    raw_data: &[u8],
    num_elements: u64,
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

/// One element of a variable-length string dataset/attribute, read as exact
/// heap bytes rather than a lossily-decoded `String`.
///
/// `None` is a *null* reference (length 0 with an undefined or zero heap
/// address), which the HDF5 model distinguishes from an empty string. `Some`
/// is a real heap object, carrying its exact bytes (possibly empty, possibly
/// containing embedded NULs or non-UTF-8 sequences). Preserving this
/// distinction lets a faithful rewrite reproduce the source byte-for-byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VlByteObject {
    /// A null VL reference (no heap object).
    Null,
    /// A heap object holding these exact bytes.
    Bytes(Vec<u8>),
}

/// Resolve a VL element's exact heap bytes from a random-access source,
/// preserving the null-vs-empty distinction and never lossily decoding.
///
/// This mirrors [`visit_vl_strings_from_source`] but yields raw bytes (and a
/// null marker) instead of a `&str`, so a faithful rewrite can reproduce
/// embedded-NUL and non-UTF-8 payloads exactly.
///
/// `element_size` is the byte width of one base-type element of the sequence.
/// For VL strings the base type is a single byte, so `element_size == 1` and the
/// reference's stored `length` (an element count) equals the byte count. For a
/// non-string VL sequence (e.g. `H5T_VLEN { H5T_NATIVE_DOUBLE }`) the stored
/// `length` counts base-type elements, so the heap object holds
/// `length * element_size` bytes — exactly what is read here.
pub(crate) fn read_vl_byte_objects_from_source<S: Source + ?Sized>(
    source: &S,
    raw_data: &[u8],
    num_elements: u64,
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
    num_elements: u64,
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

/// Resolve VL strings from an in-memory buffer by looking up each element in the
/// global heap. A thin convenience wrapper over
/// [`read_vl_strings_from_source`] used by the unit tests; production callers go
/// straight to the source-based reader so a streaming backend works unchanged.
#[cfg(test)]
pub fn read_vl_strings(
    file_data: &[u8],
    raw_data: &[u8],
    num_elements: u64,
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
mod tests {
    #[cfg(feature = "std")]
    use std::io::Cursor;

    use rstest::rstest;
    use test_util::bytes;
    use test_util::global_heap;
    use test_util::widths::Widths;

    use super::*;
    #[cfg(feature = "std")]
    use crate::source::ReadSeekSource;
    #[cfg(feature = "std")]
    use crate::source::ValidatedSource;
    use crate::width::LengthWidth;

    const FORMAT_WIDTHS: FormatWidths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };

    fn format_widths(lengths: usize) -> FormatWidths {
        let lengths = match lengths {
            2 => LengthWidth::Two,
            4 => LengthWidth::Four,
            8 => LengthWidth::Eight,
            other => panic!("unsupported test length width {other}"),
        };
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths,
        }
    }

    /// Build a global heap collection at given offset in a file buffer.
    /// Writes a global heap collection of `(index, data)` objects at `offset`,
    /// growing `file_data` to hold it.
    fn build_gcol_at(file_data: &mut Vec<u8>, offset: usize, objects: &[(u16, &[u8])]) {
        let objects: Vec<_> = objects
            .iter()
            .map(|&(index, data)| global_heap::Object::new(index, data))
            .collect();
        let collection = global_heap::collection(&objects, WIDTHS);
        file_data.resize(file_data.len().max(offset + collection.len()), 0);
        file_data[offset..offset + collection.len()].copy_from_slice(&collection);
    }

    /// The raw bytes of one variable-length element per string, each pointing
    /// at the collection at `collection_address` and an object index counting
    /// up from `start_index`.
    fn build_vl_refs(
        strings: &[&str],
        collection_address: u64,
        start_index: u16,
        widths: Widths,
    ) -> Vec<u8> {
        strings
            .iter()
            .enumerate()
            .flat_map(|(index, string)| {
                global_heap::reference(
                    string.len() as u32,
                    collection_address,
                    u32::from(start_index) + index as u32,
                    widths,
                )
            })
            .collect()
    }

    #[test]
    fn parse_vl_references_two_elements() {
        let raw = build_vl_refs(&["hello", "world"], 0x1000, 1, WIDTHS);
        let refs = parse_vl_references(&raw, 2, OffsetWidth::Eight).unwrap();
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].length, 5);
        assert_eq!(refs[0].collection_address, StoredAddress::new(0x1000));
        assert_eq!(refs[0].object_index, 1);
        assert_eq!(refs[1].length, 5);
        assert_eq!(refs[1].object_index, 2);
    }

    #[test]
    fn read_vl_strings_from_heap() {
        let gcol_offset = 256usize;
        let mut file_data = vec![0u8; 512];
        build_gcol_at(&mut file_data, gcol_offset, &[(1, b"Alice"), (2, b"Bob")]);

        let raw = build_vl_refs(&["Alice", "Bob"], gcol_offset as u64, 1, WIDTHS);
        let strings = read_vl_strings(&file_data, &raw, 2, FORMAT_WIDTHS).unwrap();
        assert_eq!(strings, vec!["Alice", "Bob"]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn read_vl_strings_from_seekable_source() {
        let gcol_offset = 256usize;
        let mut file_data = vec![0u8; 512];
        build_gcol_at(&mut file_data, gcol_offset, &[(1, b"Alice"), (2, b"Bob")]);
        let raw = build_vl_refs(&["Alice", "Bob"], gcol_offset as u64, 1, WIDTHS);
        let source = ReadSeekSource::new(Cursor::new(file_data)).unwrap();

        let strings = read_vl_strings_from_source(
            &source,
            &raw,
            2,
            FORMAT_WIDTHS,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            VlenStringReadOptions::default(),
        )
        .unwrap();
        assert_eq!(strings, vec!["Alice", "Bob"]);
    }

    #[test]
    fn null_vl_element_empty_string() {
        // length=0, address=undefined
        let mut raw = Vec::new();
        raw.extend_from_slice(&0u32.to_le_bytes()); // length=0
        raw.extend_from_slice(&u64::MAX.to_le_bytes()); // undefined address
        raw.extend_from_slice(&0u32.to_le_bytes()); // index

        let file_data = vec![0u8; 16];
        let strings = read_vl_strings(&file_data, &raw, 1, FORMAT_WIDTHS).unwrap();
        assert_eq!(strings, vec![""]);
    }

    #[test]
    fn null_vl_element_zero_address() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&0u32.to_le_bytes());
        raw.extend_from_slice(&0u64.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());

        let file_data = vec![0u8; 16];
        let strings = read_vl_strings(&file_data, &raw, 1, FORMAT_WIDTHS).unwrap();
        assert_eq!(strings, vec![""]);
    }

    #[test]
    fn parse_vl_references_truncated_error() {
        let raw = vec![0u8; 10]; // too short for 1 element with offset_size=8
        let err = parse_vl_references(&raw, 1, OffsetWidth::Eight).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[rstest]
    #[case::one_object(4096, &[(1, b"alpha".as_slice())], &[(1, 5)], &[b"alpha".as_slice()])]
    #[case::an_empty_object(4096, &[(1, b"".as_slice())], &[(1, 0)], &[b"".as_slice()])]
    #[case::a_prefix_of_an_object(4096, &[(1, b"alpha".as_slice())], &[(1, 3)], &[b"alp".as_slice()])]
    #[case::free_space_ending_at_the_collection_end(
        4096,
        &[(1, b"alpha".as_slice()), (2, b"beta".as_slice())],
        &[(2, 4), (1, 5)],
        &[b"beta".as_slice(), b"alpha".as_slice()]
    )]
    #[case::one_alignment_past_the_minimum_size(4104, &[(1, b"alpha".as_slice())], &[(1, 5)], &[b"alpha".as_slice()])]
    #[case::a_tail_shorter_than_an_object_header(
        4096,
        &[(1, [9u8; 4056].as_slice())],
        &[(1, 4056)],
        &[[9u8; 4056].as_slice()]
    )]
    fn a_read_returns_the_requested_objects(
        #[values(2, 4, 8)] lengths: usize,
        #[case] size: usize,
        #[case] objects: &[(u16, &[u8])],
        #[case] requests: &[(u16, u32)],
        #[case] expected: &[&[u8]],
    ) {
        assert_every_reader_returns(
            &collection_file(lengths, Some(size), objects, Patch::None),
            &references(requests),
            lengths,
            BaseAddress::ZERO,
            Ok(expected.iter().copied().map(payload).collect()),
        );
    }

    #[rstest]
    #[case::every_object(&(1..=2000).collect::<Vec<u16>>())]
    #[case::a_sparse_lookup(&[1500, 7, 2000])]
    fn a_read_finds_objects_among_many(
        #[values(2, 4, 8)] lengths: usize,
        #[case] requested: &[u16],
    ) {
        let payloads: Vec<[u8; 2]> = (1..=2000u16).map(u16::to_le_bytes).collect();
        let objects: Vec<(u16, &[u8])> = (1..)
            .zip(payloads.iter().map(<[u8; 2]>::as_slice))
            .collect();
        let requests: Vec<(u16, u32)> = requested.iter().map(|&index| (index, 2)).collect();
        assert_every_reader_returns(
            &collection_file(lengths, None, &objects, Patch::None),
            &references(&requests),
            lengths,
            BaseAddress::ZERO,
            Ok(requested
                .iter()
                .map(|index| payload(&index.to_le_bytes()))
                .collect()),
        );
    }

    #[rstest]
    fn a_read_returns_an_object_larger_than_a_two_byte_length(#[values(4, 8)] lengths: usize) {
        let large = vec![7; 70_000];
        assert_every_reader_returns(
            &collection_file(lengths, None, &[(1, &large)], Patch::None),
            &references(&[(1, 70_000)]),
            lengths,
            BaseAddress::ZERO,
            Ok(vec![payload(&large)]),
        );
    }

    // Byte 8 holds the collection size, and byte 48 the size field of the second object header.
    #[rstest]
    #[case::a_collection_size_below_the_minimum(
        &[(1, b"alpha".as_slice())],
        Patch::Length { at: 8, value: 4095 },
        1,
        FormatError::GlobalHeapCollectionTooSmall { collection_address: 512, size: 4095 }
    )]
    #[case::an_unaligned_collection_size(
        &[(1, b"alpha".as_slice())],
        Patch::Length { at: 8, value: 4100 },
        1,
        FormatError::UnalignedGlobalHeapCollection { collection_address: 512, size: 4100 }
    )]
    #[case::a_duplicate_index_after_the_requested_one(
        &[(1, b"first".as_slice()), (2, b"second".as_slice()), (1, b"again".as_slice())],
        Patch::None,
        1,
        FormatError::DuplicateGlobalHeapObjectIndex { collection_address: 512, index: 1 }
    )]
    #[case::bytes_after_the_free_space(
        &[(1, b"alpha".as_slice())],
        Patch::Length { at: 48, value: 4048 },
        1,
        FormatError::MisplacedGlobalHeapFreeSpace {
            address: 552,
            size: 4048,
            collection_end: 4608,
        }
    )]
    #[case::a_nonzero_index_in_a_short_tail(
        &[(1, [9u8; 4056].as_slice())],
        Patch::Index { at: 4088, value: 7 },
        1,
        FormatError::GlobalHeapObjectOutsideCollection {
            address: 4600,
            index: 7,
            collection_end: 4608,
        }
    )]
    #[case::a_final_object_past_the_collection_end(
        &[(1, b"first".as_slice()), (2, b"second".as_slice())],
        Patch::Length { at: 48, value: 4041 },
        1,
        FormatError::GlobalHeapObjectOutsideCollection {
            address: 552,
            index: 2,
            collection_end: 4608,
        }
    )]
    #[case::an_absent_index(
        &[(1, b"first".as_slice())],
        Patch::None,
        2,
        FormatError::GlobalHeapObjectNotFound { collection_address: 512, index: 2 }
    )]
    #[case::index_zero(
        &[(1, b"first".as_slice())],
        Patch::None,
        0,
        FormatError::GlobalHeapObjectNotFound { collection_address: 512, index: 0 }
    )]
    fn a_read_rejects_a_malformed_collection_or_an_absent_object(
        #[values(2, 4, 8)] lengths: usize,
        #[case] objects: &[(u16, &[u8])],
        #[case] patch: Patch,
        #[case] index: u16,
        #[case] expected: FormatError,
    ) {
        assert_every_reader_returns(
            &collection_file(lengths, Some(4096), objects, patch),
            &references(&[(index, 1)]),
            lengths,
            BaseAddress::ZERO,
            Err(expected),
        );
    }

    #[test]
    fn a_collection_past_the_end_of_the_source_is_an_unexpected_eof() {
        let mut file = collection_file(8, None, &[(1, b"alpha")], Patch::None);
        file.truncate(4600);
        assert_every_reader_returns(
            &file,
            &references(&[(1, 5)]),
            8,
            BaseAddress::ZERO,
            Err(FormatError::UnexpectedSourceEof {
                expected: 4608,
                available: 4600,
            }),
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_short_source_read_is_a_source_error() {
        /// A source whose metadata reads return one byte fewer than the caller requests.
        struct ShortMetadata(Vec<u8>);

        impl Source for ShortMetadata {
            fn len(&self) -> u64 {
                self.0.len() as u64
            }

            fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
                BytesSource::new(&self.0).read_at(offset, buf)
            }

            fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
                BytesSource::new(&self.0).read_exact_at(offset, len - 1)
            }
        }

        let file = collection_file(8, None, &[(1, b"alpha")], Patch::None);
        let refs = references(&[(1, 5)]);
        assert_eq!(
            read_vl_byte_objects_from_source(
                &ValidatedSource::new(ShortMetadata(file)),
                &refs,
                1,
                FORMAT_WIDTHS,
                AddressContext::from_effective_base(BaseAddress::ZERO),
                1,
                VlenStringReadOptions::default(),
            )
            .unwrap_err(),
            FormatError::Source(
                "the source returned 15 bytes for a 16-byte read at offset 512".into()
            )
        );
    }

    // The rejection reports the physical address of the collection, 512 + 1024.
    #[rstest]
    #[case::read(4096, Ok(vec![payload(b"alpha")]))]
    #[case::rejected(
        4095,
        Err(FormatError::GlobalHeapCollectionTooSmall { collection_address: 1536, size: 4095 })
    )]
    fn a_userblock_shifts_the_collection_by_the_base(
        #[case] size: u64,
        #[case] expected: Result<Vec<VlByteObject>, FormatError>,
    ) {
        let mut collection =
            global_heap::collection(&[global_heap::Object::new(1, b"alpha")], WIDTHS);
        bytes::set_uint_at(&mut collection, 8, 8, size);
        let mut file = vec![0; USERBLOCK + 1024];
        file.extend_from_slice(&collection);
        let refs = global_heap::reference(5, 1024, 1, WIDTHS);
        assert_every_reader_returns(
            &file,
            &refs,
            8,
            BaseAddress::new(USERBLOCK as u64),
            expected,
        );
    }

    #[test]
    fn a_collection_above_four_gib_is_read_at_its_u64_position() {
        /// A source that holds `bytes` from the position `start` on.
        struct Sparse {
            start: u64,
            bytes: Vec<u8>,
        }

        impl Source for Sparse {
            fn len(&self) -> u64 {
                self.start + self.bytes.len() as u64
            }

            fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
                let at = offset.checked_sub(self.start).unwrap();
                BytesSource::new(&self.bytes).read_at(at, buf)
            }
        }

        let address = (1 << 32) + 4096;
        let source = Sparse {
            start: address,
            bytes: global_heap::collection(
                &[
                    global_heap::Object::new(1, b"alpha"),
                    global_heap::Object::new(2, b"beta"),
                ],
                WIDTHS,
            ),
        };
        let refs: Vec<u8> = [(5, 1), (4, 2)]
            .into_iter()
            .flat_map(|(len, index)| global_heap::reference(len, address, index, WIDTHS))
            .collect();
        assert_eq!(
            read_vl_byte_objects_from_source(
                &source,
                &refs,
                2,
                FORMAT_WIDTHS,
                AddressContext::from_effective_base(BaseAddress::ZERO),
                1,
                VlenStringReadOptions::default(),
            )
            .unwrap(),
            vec![payload(b"alpha"), payload(b"beta")]
        );
    }

    /// Returns a file that holds a collection of `objects` at byte [`COLLECTION`], with length
    /// fields `lengths` bytes wide and `patch` applied.
    ///
    /// The collection is `size` bytes, or the size [`global_heap::collection`] gives it where
    /// `size` is `None`.
    fn collection_file(
        lengths: usize,
        size: Option<usize>,
        objects: &[(u16, &[u8])],
        patch: Patch,
    ) -> Vec<u8> {
        let objects: Vec<_> = objects
            .iter()
            .map(|&(index, data)| global_heap::Object::new(index, data))
            .collect();
        let widths = Widths::new(8, lengths);
        let mut collection = match size {
            Some(size) => global_heap::collection_of_size(&objects, widths, size),
            None => global_heap::collection(&objects, widths),
        };
        match patch {
            Patch::None => {}
            Patch::Index { at, value } => bytes::set_u16_at(&mut collection, at, value),
            Patch::Length { at, value } => {
                bytes::set_uint_at(&mut collection, at, lengths, value);
            }
        }
        let mut file = vec![0; COLLECTION];
        file.extend_from_slice(&collection);
        file
    }

    /// Returns the references to the `(index, length)` pairs of `requests`, in the collection at
    /// [`COLLECTION`].
    fn references(requests: &[(u16, u32)]) -> Vec<u8> {
        requests
            .iter()
            .flat_map(|&(index, len)| {
                global_heap::reference(len, COLLECTION as u64, u32::from(index), WIDTHS)
            })
            .collect()
    }

    /// Asserts that [`read_vl_byte_objects_from_source`] returns `expected` for `refs` in `file`,
    /// and [`read_vl_strings_from_source`] `expected` decoded as lossy UTF-8.
    ///
    /// The assertions read `file` through a [`BytesSource`] and, with `std`, a [`ReadSeekSource`].
    fn assert_every_reader_returns(
        file: &[u8],
        refs: &[u8],
        lengths: usize,
        base: BaseAddress,
        expected: Result<Vec<VlByteObject>, FormatError>,
    ) {
        let count = (refs.len() / VL_REF_BYTES) as u64;
        let widths = format_widths(lengths);
        let addresses = AddressContext::from_effective_base(base);
        let options = VlenStringReadOptions::default();
        let objects = |source: &dyn Source| {
            read_vl_byte_objects_from_source(source, refs, count, widths, addresses, 1, options)
        };
        let strings = |source: &dyn Source| {
            read_vl_strings_from_source(source, refs, count, widths, addresses, options)
        };
        let decoded = expected.clone().map(|objects| {
            objects
                .iter()
                .map(|object| match object {
                    VlByteObject::Null => String::new(),
                    VlByteObject::Bytes(bytes) => String::from_utf8_lossy(bytes).into_owned(),
                })
                .collect::<Vec<_>>()
        });
        let buffered = BytesSource::new(file);
        assert_eq!(objects(&buffered), expected);
        assert_eq!(strings(&buffered), decoded);
        #[cfg(feature = "std")]
        {
            let streamed = ReadSeekSource::new(Cursor::new(file.to_vec())).unwrap();
            assert_eq!(objects(&streamed), expected);
            assert_eq!(strings(&streamed), decoded);
        }
    }

    fn payload(data: &[u8]) -> VlByteObject {
        VlByteObject::Bytes(data.to_vec())
    }

    /// A change to the bytes of a collection: the object index or the length field at the offset
    /// `at` from its start, set to `value`.
    enum Patch {
        Index { at: usize, value: u16 },
        Length { at: usize, value: u64 },
        None,
    }

    /// The address of the collection in a file [`collection_file`] returns.
    const COLLECTION: usize = 512;

    /// The size of the userblock in the file of [`a_userblock_shifts_the_collection_by_the_base`].
    const USERBLOCK: usize = 512;

    const WIDTHS: Widths = Widths::EIGHT;
}

#[cfg(test)]
mod embedded_slot_tests {
    use super::*;
    use crate::{
        DatatypeByteOrder,
        datatype::{CompoundMember, StringPadding, layout::FixedPointLayout},
    };

    fn vlen_string() -> Datatype {
        Datatype::VariableLength {
            is_string: true,
            base_type: Box::new(Datatype::String {
                size: 1,
                charset: CharacterSet::Utf8,
                padding: StringPadding::NullTerminate,
            }),
            padding: Some(StringPadding::NullTerminate),
            charset: Some(CharacterSet::Utf8),
        }
    }

    fn vlen_i32_sequence() -> Datatype {
        Datatype::VariableLength {
            is_string: false,
            base_type: Box::new(Datatype::FixedPoint {
                size: 4,
                byte_order: DatatypeByteOrder::LittleEndian,
                layout: FixedPointLayout {
                    signed: true,
                    bit_offset: 0,
                    bit_precision: 32,
                },
            }),
            padding: None,
            charset: None,
        }
    }

    fn i32_type() -> Datatype {
        Datatype::FixedPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FixedPointLayout {
                signed: true,
                bit_offset: 0,
                bit_precision: 32,
            },
        }
    }

    fn member(name: &str, byte_offset: u64, datatype: Datatype) -> CompoundMember {
        crate::datatype::__private::compound_member(name.to_string(), byte_offset, datatype)
    }

    #[test]
    fn a_plain_datatype_reaches_no_reference() {
        assert_eq!(embedded_vlen_slots(&i32_type()), Some(Vec::new()));
    }

    #[test]
    fn a_top_level_vlen_is_its_own_slot() {
        assert_eq!(
            embedded_vlen_slots(&vlen_string()),
            Some(vec![EmbeddedVlSlot {
                byte_offset: 0,
                element_size: 1
            }])
        );
    }

    /// Each member keeps its own base-type width: a string reference counts
    /// bytes, a sequence reference counts base-type elements. Collapsing the two
    /// would read the wrong number of heap bytes back.
    #[test]
    fn compound_members_keep_their_own_element_size() {
        let dt = Datatype::Compound {
            size: 40,
            members: vec![
                member("label", 0, vlen_string()),
                member("id", 16, i32_type()),
                member("samples", 24, vlen_i32_sequence()),
            ],
        };
        assert_eq!(
            embedded_vlen_slots(&dt),
            Some(vec![
                EmbeddedVlSlot {
                    byte_offset: 0,
                    element_size: 1
                },
                EmbeddedVlSlot {
                    byte_offset: 24,
                    element_size: 4
                },
            ])
        );
    }

    /// A compound nested inside a compound, and an array of compounds, both reach
    /// references that a single-level walk would miss.
    #[test]
    fn nested_compounds_and_arrays_are_walked() {
        let inner = Datatype::Compound {
            size: 20,
            members: vec![member("id", 0, i32_type()), member("s", 4, vlen_string())],
        };
        let nested = Datatype::Compound {
            size: 24,
            members: vec![
                member("n", 0, i32_type()),
                member("inner", 4, inner.clone()),
            ],
        };
        assert_eq!(
            embedded_vlen_slots(&nested),
            Some(vec![EmbeddedVlSlot {
                byte_offset: 8,
                element_size: 1
            }])
        );

        let array = Datatype::Array {
            base_type: Box::new(inner),
            dimensions: vec![3],
        };
        assert_eq!(
            embedded_vlen_slots(&array),
            Some(vec![
                EmbeddedVlSlot {
                    byte_offset: 4,
                    element_size: 1
                },
                EmbeddedVlSlot {
                    byte_offset: 24,
                    element_size: 1
                },
                EmbeddedVlSlot {
                    byte_offset: 44,
                    element_size: 1
                },
            ])
        );
    }

    /// A datatype whose declared size cannot hold the references it declares is
    /// rejected rather than walked: the element bytes and the datatype disagree,
    /// so any offset derived from the latter would index the wrong place.
    #[test]
    fn a_datatype_too_small_for_its_own_references_is_rejected() {
        let dt = Datatype::Compound {
            size: 8, // one VL reference needs 16
            members: vec![member("s", 0, vlen_string())],
        };
        assert_eq!(embedded_vlen_slots(&dt), None);
    }

    /// An array declaring dimensions whose product overflows must not drive an
    /// unbounded walk; the capacity bound stops it and the result is rejected.
    #[test]
    fn an_array_declaring_absurd_dimensions_is_rejected_not_walked() {
        let dt = Datatype::Array {
            base_type: Box::new(vlen_string()),
            dimensions: vec![u32::MAX, u32::MAX],
        };
        assert_eq!(embedded_vlen_slots(&dt), None);
    }

    /// Deeply nested arrays must cost time proportional to the slots they
    /// produce, not exponential in nesting depth.
    ///
    /// Walking each entry from scratch recomputes the same sub-tree once per
    /// entry at every level, so a datatype of ~13 bytes per level — trivially
    /// small in an object header, and depth-unlimited in `Datatype::parse` — used
    /// to burn seconds here and hours a few levels further down. Any repack of an
    /// untrusted file reaches this walk.
    #[test]
    fn deeply_nested_arrays_cost_time_proportional_to_their_slots() {
        let mut dt = vlen_string();
        for _ in 0..17 {
            dt = Datatype::Array {
                base_type: Box::new(dt),
                dimensions: vec![2],
            };
        }
        let started = std::time::Instant::now();
        let slots = embedded_vlen_slots(&dt).expect("a well-formed nesting must be walkable");
        let elapsed = started.elapsed();

        assert_eq!(slots.len(), 1 << 17, "one slot per leaf entry");
        // The linear walk does this in milliseconds even unoptimized; the
        // exponential one took seconds in release. A wide margin keeps the test
        // from being a CI-timing flake while still failing the regression by
        // orders of magnitude.
        assert!(
            elapsed < std::time::Duration::from_secs(10),
            "walking {} slots took {elapsed:?}; the walk is no longer linear in its output",
            slots.len()
        );
    }

    /// A member offset is read from the file as a `u64`, so it can name a
    /// position no `usize` can address. Truncating it would land the slot
    /// somewhere plausible *inside* the element, where the size check below would
    /// wave it through and the rewrite would corrupt an unrelated field; the walk
    /// has to refuse instead. On 64-bit the offset simply exceeds the element.
    #[test]
    fn a_member_offset_beyond_the_address_space_is_rejected() {
        let dt = Datatype::Compound {
            size: 40,
            members: vec![member("s", u64::MAX - 8, vlen_string())],
        };
        assert_eq!(embedded_vlen_slots(&dt), None);
    }
}
