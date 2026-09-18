//! Variable-length data reading (VL strings & VL sequences).
//!
//! VL data elements in HDF5 store their values in the global heap.
//! The raw data for each element contains a global heap ID:
//! `sequence_length(4 LE) + collection_address(offset_size LE) + object_index(4 LE)`.

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};

use crate::address::{BaseAddress, StoredAddress};
use crate::bytes::read_offset;
use crate::convert::{Narrow, is_undefined_addr};
use crate::datatype::{CharacterSet, Datatype};
use crate::error::FormatError;
use crate::global_heap::GlobalHeapIndex;
#[cfg(test)]
use crate::source::BytesSource;
use crate::source::Source;

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

/// Parse VL global heap references from raw attribute/dataset data.
pub fn parse_vl_references(
    raw_data: &[u8],
    num_elements: u64,
    offset_size: u8,
) -> Result<Vec<VlElement>, FormatError> {
    let elem_size = 4 + offset_size as u64 + 4; // length + address + index
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
    let mut pos = 0;

    for _ in 0..num_elements {
        let length = u32::from_le_bytes([
            raw_data[pos],
            raw_data[pos + 1],
            raw_data[pos + 2],
            raw_data[pos + 3],
        ]);
        pos += 4;

        let collection_address = StoredAddress::new(read_offset(raw_data, pos, offset_size)?);
        pos += offset_size as usize;

        let object_index = u32::from_le_bytes([
            raw_data[pos],
            raw_data[pos + 1],
            raw_data[pos + 2],
            raw_data[pos + 3],
        ]);
        pos += 4;

        elements.push(VlElement {
            length,
            collection_address,
            object_index,
        });
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
    offset_size: u8,
) -> Result<u64, FormatError> {
    check_element_limit(num_elements, VlenStringReadOptions::default())?;
    let refs = parse_vl_references(raw_data, num_elements, offset_size)?;
    payload_size(&refs, VlenStringReadOptions::default())
}

/// Resolve VL strings from a random-access file source and pass them to a
/// visitor one at a time.
pub fn visit_vl_strings_from_source<S, F>(
    source: &S,
    raw_data: &[u8],
    num_elements: u64,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
    options: VlenStringReadOptions,
    mut visitor: F,
) -> Result<(), FormatError>
where
    S: Source + ?Sized,
    F: FnMut(&str),
{
    check_element_limit(num_elements, options)?;
    let refs = parse_vl_references(raw_data, num_elements, offset_size)?;
    payload_size(&refs, options)?;

    // This call's object indices per (base-adjusted) collection address, so
    // each collection's directory walk retains only the entries the call
    // resolves — a row window of a large dataset would otherwise be charged
    // the full directory of every touched collection (a writer may pack every
    // string into one collection). Grouping collects only references the
    // resolve loop below will look up; invalid ones surface their errors
    // there, in element order.
    let mut wanted: Vec<(u64, Vec<u16>)> = Vec::new();
    for element in &refs {
        if is_undefined_addr(element.collection_address.get(), offset_size)
            || (element.length == 0 && element.collection_address.get() == 0)
        {
            continue;
        }
        let Ok(address) = base_address.absolute(element.collection_address) else {
            continue;
        };
        let Ok(index) = u16::try_from(element.object_index) else {
            continue;
        };
        match wanted.binary_search_by_key(&address, |&(a, _)| a) {
            Ok(pos) => wanted[pos].1.push(index),
            Err(pos) => wanted.insert(pos, (address, Vec::from([index]))),
        }
    }
    for (_, indices) in &mut wanted {
        indices.sort_unstable();
    }

    let mut collections: Vec<(u64, GlobalHeapIndex)> = Vec::new();
    for element in &refs {
        if element.length == 0
            && (is_undefined_addr(element.collection_address.get(), offset_size)
                || element.collection_address.get() == 0)
        {
            visitor("");
            continue;
        }
        if is_undefined_addr(element.collection_address.get(), offset_size) {
            return Err(FormatError::VlDataError(
                "non-empty VL element has an undefined heap address".into(),
            ));
        }

        let collection_address = base_address.absolute(element.collection_address)?;
        let collection_pos = match collections
            .iter()
            .position(|(address, _)| *address == collection_address)
        {
            Some(pos) => pos,
            None => {
                let keep = wanted
                    .binary_search_by_key(&collection_address, |&(a, _)| a)
                    .map(|pos| wanted[pos].1.as_slice())
                    .unwrap_or(&[]);
                let collection = GlobalHeapIndex::parse_filtered(
                    source,
                    collection_address,
                    length_size,
                    |i| keep.binary_search(&i).is_ok(),
                )?;
                collections.push((collection_address, collection));
                collections.len() - 1
            }
        };

        let index: u16 = element.object_index.narrow_or_else(|| {
            FormatError::VlDataError(format!(
                "global heap object index {} does not fit u16",
                element.object_index
            ))
        })?;
        let object = collections[collection_pos].1.get_object(index).ok_or(
            FormatError::GlobalHeapObjectNotFound {
                collection_address,
                index,
            },
        )?;
        if u64::from(element.length) > object.size {
            return Err(FormatError::VlDataError(format!(
                "VL element length {} exceeds global heap object size {}",
                element.length, object.size
            )));
        }

        let bytes = source.read_exact_at(object.data_address, element.length as usize)?;
        let string = String::from_utf8_lossy(&bytes);
        visitor(&string);
    }

    Ok(())
}

/// Resolve VL strings from a random-access file source.
pub fn read_vl_strings_from_source<S: Source + ?Sized>(
    source: &S,
    raw_data: &[u8],
    num_elements: u64,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
    options: VlenStringReadOptions,
) -> Result<Vec<String>, FormatError> {
    let mut strings = Vec::new();
    visit_vl_strings_from_source(
        source,
        raw_data,
        num_elements,
        offset_size,
        length_size,
        base_address,
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
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
    element_size: usize,
    options: VlenStringReadOptions,
) -> Result<Vec<VlByteObject>, FormatError> {
    check_element_limit(num_elements, options)?;
    let refs = parse_vl_references(raw_data, num_elements, offset_size)?;
    payload_size(&refs, options)?;

    let mut objects = Vec::with_capacity(refs.len());
    let mut collections: Vec<(u64, GlobalHeapIndex)> = Vec::new();
    for element in &refs {
        if element.length == 0
            && (is_undefined_addr(element.collection_address.get(), offset_size)
                || element.collection_address.get() == 0)
        {
            objects.push(VlByteObject::Null);
            continue;
        }
        if is_undefined_addr(element.collection_address.get(), offset_size) {
            return Err(FormatError::VlDataError(
                "non-empty VL element has an undefined heap address".into(),
            ));
        }

        let collection_address = base_address.absolute(element.collection_address)?;
        let collection_pos = match collections
            .iter()
            .position(|(address, _)| *address == collection_address)
        {
            Some(pos) => pos,
            None => {
                let collection = GlobalHeapIndex::parse(source, collection_address, length_size)?;
                collections.push((collection_address, collection));
                collections.len() - 1
            }
        };

        let index: u16 = element.object_index.narrow_or_else(|| {
            FormatError::VlDataError(format!(
                "global heap object index {} does not fit u16",
                element.object_index
            ))
        })?;
        let object = collections[collection_pos].1.get_object(index).ok_or(
            FormatError::GlobalHeapObjectNotFound {
                collection_address,
                index,
            },
        )?;
        // The heap object holds `length` base-type elements of `element_size`
        // bytes each. Compute the byte count with checked arithmetic so a hostile
        // `length` cannot overflow, and bound it by the heap object's own size.
        let byte_len = (element.length as u64)
            .checked_mul(element_size as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset: u64::from(element.length),
                length: element_size as u64,
            })?;
        if byte_len > object.size {
            return Err(FormatError::VlDataError(format!(
                "VL element length {} ({} bytes) exceeds global heap object size {}",
                element.length, byte_len, object.size
            )));
        }

        let bytes = source.read_exact_at(object.data_address, byte_len.to_usize()?)?;
        objects.push(VlByteObject::Bytes(bytes));
    }

    Ok(objects)
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
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<String>, FormatError> {
    read_vl_strings_from_source(
        &BytesSource::new(file_data),
        raw_data,
        num_elements,
        offset_size,
        length_size,
        BaseAddress::ZERO,
        VlenStringReadOptions::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a global heap collection at given offset in a file buffer.
    fn build_gcol_at(
        file_data: &mut Vec<u8>,
        offset: usize,
        objects: &[(u16, &[u8])], // (index, data)
    ) {
        let length_size = 8usize;

        // Ensure file_data is large enough
        let header_size = 8 + length_size;
        let mut obj_total = 0usize;
        for (_, data) in objects {
            let padded = (data.len() + 7) & !7;
            obj_total += 8 + length_size + padded;
        }
        obj_total += 2; // free space marker
        let collection_size = header_size + obj_total;
        let needed = offset + collection_size;
        if file_data.len() < needed {
            file_data.resize(needed, 0);
        }

        let mut pos = offset;
        // Signature
        file_data[pos..pos + 4].copy_from_slice(b"GCOL");
        file_data[pos + 4] = 1; // version
        // reserved(3) already 0
        pos += 8;
        file_data[pos..pos + 8].copy_from_slice(&(collection_size as u64).to_le_bytes());
        pos += 8;

        for (index, data) in objects {
            file_data[pos..pos + 2].copy_from_slice(&index.to_le_bytes());
            file_data[pos + 2..pos + 4].copy_from_slice(&1u16.to_le_bytes()); // ref_count
            // reserved(4) already 0
            pos += 8;
            file_data[pos..pos + 8].copy_from_slice(&(data.len() as u64).to_le_bytes());
            pos += 8;
            file_data[pos..pos + data.len()].copy_from_slice(data);
            let padded = (data.len() + 7) & !7;
            pos += padded;
        }
        // free space marker
        file_data[pos..pos + 2].copy_from_slice(&0u16.to_le_bytes());
    }

    /// Build VL reference raw data for given strings at a collection address.
    fn build_vl_refs(
        strings: &[&str],
        collection_address: u64,
        start_index: u16,
        offset_size: u8,
    ) -> Vec<u8> {
        let mut raw = Vec::new();
        for (i, s) in strings.iter().enumerate() {
            raw.extend_from_slice(&(s.len() as u32).to_le_bytes());
            match offset_size {
                4 => raw.extend_from_slice(&(collection_address as u32).to_le_bytes()),
                8 => raw.extend_from_slice(&collection_address.to_le_bytes()),
                _ => panic!("unsupported"),
            }
            raw.extend_from_slice(&(start_index as u32 + i as u32).to_le_bytes());
        }
        raw
    }

    #[test]
    fn parse_vl_references_two_elements() {
        let raw = build_vl_refs(&["hello", "world"], 0x1000, 1, 8);
        let refs = parse_vl_references(&raw, 2, 8).unwrap();
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

        let raw = build_vl_refs(&["Alice", "Bob"], gcol_offset as u64, 1, 8);
        let strings = read_vl_strings(&file_data, &raw, 2, 8, 8).unwrap();
        assert_eq!(strings, vec!["Alice", "Bob"]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn read_vl_strings_from_seekable_source() {
        use std::io::Cursor;

        use crate::source::ReadSeekSource;

        let gcol_offset = 256usize;
        let mut file_data = vec![0u8; 512];
        build_gcol_at(&mut file_data, gcol_offset, &[(1, b"Alice"), (2, b"Bob")]);
        let raw = build_vl_refs(&["Alice", "Bob"], gcol_offset as u64, 1, 8);
        let source = ReadSeekSource::new(Cursor::new(file_data)).unwrap();

        let strings = read_vl_strings_from_source(
            &source,
            &raw,
            2,
            8,
            8,
            BaseAddress::ZERO,
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
        let strings = read_vl_strings(&file_data, &raw, 1, 8, 8).unwrap();
        assert_eq!(strings, vec![""]);
    }

    #[test]
    fn null_vl_element_zero_address() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&0u32.to_le_bytes());
        raw.extend_from_slice(&0u64.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());

        let file_data = vec![0u8; 16];
        let strings = read_vl_strings(&file_data, &raw, 1, 8, 8).unwrap();
        assert_eq!(strings, vec![""]);
    }

    #[test]
    fn parse_vl_references_truncated_error() {
        let raw = vec![0u8; 10]; // too short for 1 element with offset_size=8
        let err = parse_vl_references(&raw, 1, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }
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
        CompoundMember {
            name: name.to_string(),
            byte_offset,
            datatype,
        }
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
