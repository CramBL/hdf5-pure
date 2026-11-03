//! Conversion of variable-length references from eight-byte heap addresses in place.

use alloc::boxed::Box;
use alloc::vec::Vec;

use byteorder::ByteOrder;
use byteorder::LittleEndian;

use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::Datatype;
use hdf5_pure_core::StoredAddress;

use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// Compacts variable-length references in `raw` to `offset_width`-byte heap addresses.
///
/// The input stores eight-byte heap addresses. Converts references directly, within arrays,
/// and within compound members. Retains the buffer allocation and declared compound record
/// lengths and member offsets. With eight-byte offsets, returns without inspecting `raw`.
///
/// With `patches` set, all heap addresses must be zero. Each patch identifies a heap-backed
/// reference's starting byte. Updates these positions after compaction while retaining their
/// order. References absent from `patches` must be null. Without patches, checks each stored
/// heap address against the selected width.
///
/// # Errors
///
/// Returns [`FormatError::DataSizeMismatch`] if `raw` does not contain whole canonical elements,
/// [`FormatError::UnexpectedEof`] if a reference extends past the buffer,
/// [`FormatError::AddressTooLarge`] if a heap address exceeds `offset_width`, or
/// [`FormatError::DefinedAddressIsUndefined`] if it equals that width's undefined marker.
/// Returns [`FormatError::Internal`] for inconsistent patch positions or already resolved staged
/// addresses. Propagates zero-sized datatype, array-size, compound-field, arithmetic, and
/// host-size conversion errors. The buffer and patches may be partially modified on error.
///
/// # Examples
///
/// Converting a resolved heap address:
///
/// ```
/// use hdf5_pure_format::{
///     __private as format, CharacterSet, Datatype, FormatError, OffsetWidth, StringPadding,
///     VariableLengthType,
/// };
///
/// let datatype = Datatype::VariableLength(VariableLengthType::string(
///     StringPadding::NullTerminate,
///     CharacterSet::Utf8,
/// ));
/// // Count 1, heap address 0x1000, object index 3.
/// let mut raw = vec![1, 0, 0, 0, 0, 0x10, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0];
/// format::narrow_variable_length_references(OffsetWidth::Two, &datatype, &mut raw, None)?;
/// assert_eq!(raw, [1, 0, 0, 0, 0, 0x10, 3, 0, 0, 0]);
/// # Ok::<(), FormatError>(())
/// ```
///
/// For staged references, pass their starting offsets in `patches`. The patch order is preserved:
///
/// ```
/// use hdf5_pure_format::{
///     __private as format, CharacterSet, Datatype, FormatError, OffsetWidth, StringPadding,
///     VariableLengthType,
/// };
///
/// let datatype = Datatype::VariableLength(VariableLengthType::string(
///     StringPadding::NullTerminate,
///     CharacterSet::Utf8,
/// ));
/// let mut raw = vec![
///     2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
///     3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
/// ];
/// let mut patches = [16, 0];
/// format::narrow_variable_length_references(
///     OffsetWidth::Two,
///     &datatype,
///     &mut raw,
///     Some(&mut patches),
/// )?;
/// assert_eq!(raw, [2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0]);
/// assert_eq!(patches, [10, 0]);
/// # Ok::<(), FormatError>(())
/// ```
pub fn narrow_variable_length_references(
    offset_width: OffsetWidth,
    datatype: &Datatype,
    raw: &mut Vec<u8>,
    mut patches: Option<&mut [usize]>,
) -> Result<(), FormatError> {
    if offset_width == OffsetWidth::Eight {
        return Ok(());
    }
    let layout = ElementLayout::new(datatype, offset_width)?;
    if !layout.has_references() {
        return Ok(());
    }
    let source_size = datatype.element_size()?.get().to_usize()?;
    if !raw.len().is_multiple_of(source_size) {
        return Err(FormatError::DataSizeMismatch {
            expected: source_size,
            actual: raw.len(),
        });
    }
    if let Some(patches) = patches.as_deref_mut() {
        for (ordinal, &offset) in patches.iter().enumerate() {
            let address_at =
                offset
                    .checked_add(SEQUENCE_LENGTH_BYTES)
                    .ok_or(FormatError::OffsetOverflow {
                        offset: offset.to_u64(),
                        length: SEQUENCE_LENGTH_BYTES.to_u64(),
                    })?;
            if bytes::read_offset_width(raw, address_at, OffsetWidth::Eight)? != 0 {
                return Err(FormatError::Internal(
                    "a staged variable-length address is already resolved".into(),
                ));
            }
            bytes::write_stored_address_at(
                raw,
                address_at,
                StoredAddress::new(ordinal.to_u64() + 1),
                OffsetWidth::Eight,
            )?;
        }
    }
    let count = raw.len() / source_size;
    let mut visited = 0;
    for element in 0..count {
        layout.write(
            raw,
            element * source_size,
            element * layout.target_size,
            offset_width,
            &mut patches,
            &mut visited,
        )?;
    }
    if patches
        .as_ref()
        .is_some_and(|patches| patches.len() != visited)
    {
        return Err(FormatError::Internal(
            "staged variable-length references do not match the datatype".into(),
        ));
    }
    raw.truncate(count * layout.target_size);
    Ok(())
}

/// The canonical and selected-width geometry of one stored value.
struct ElementLayout {
    source_size: usize,
    target_size: usize,
    kind: ElementKind,
}

impl ElementLayout {
    /// Plans reference, array, and compound positions at `offsets`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::CompoundFieldOutOfBounds`] if a member's canonical bytes exceed its
    /// record, [`FormatError::LengthTooLarge`] if an array size exceeds its four-byte field,
    /// [`FormatError::OffsetOverflow`] for array geometry beyond [`u64::MAX`], or
    /// [`FormatError::ValueTooLargeForPlatform`] if geometry exceeds the target's [`usize`] range.
    fn new(datatype: &Datatype, offsets: OffsetWidth) -> Result<Self, FormatError> {
        let (source_size, target_size, kind) = match datatype {
            Datatype::VariableLength(_) => (
                REFERENCE_BYTES,
                SEQUENCE_LENGTH_BYTES + usize::from(offsets.get()) + OBJECT_INDEX_BYTES,
                ElementKind::Reference,
            ),
            Datatype::Array {
                base_type,
                dimensions,
            } => {
                let base = Self::new(base_type, offsets)?;
                let count = dimensions.iter().try_fold(1u64, |count, &dimension| {
                    count
                        .checked_mul(u64::from(dimension))
                        .ok_or(FormatError::OffsetOverflow {
                            offset: count,
                            length: u64::from(dimension),
                        })
                })?;
                let size = |stride: usize| -> Result<usize, FormatError> {
                    let size =
                        count
                            .checked_mul(stride.to_u64())
                            .ok_or(FormatError::OffsetOverflow {
                                offset: count,
                                length: stride.to_u64(),
                            })?;
                    if size > u64::from(u32::MAX) {
                        return Err(FormatError::LengthTooLarge {
                            length: size,
                            length_size: DATATYPE_SIZE_BYTES,
                        });
                    }
                    size.to_usize()
                };
                let source_size = size(base.source_size)?;
                let target_size = size(base.target_size)?;
                let kind = if base.has_references() {
                    ElementKind::Array {
                        count: count.to_usize()?,
                        base: Box::new(base),
                    }
                } else {
                    ElementKind::Fixed
                };
                (source_size, target_size, kind)
            }
            Datatype::Compound { size, members } => {
                let mut fields = Vec::new();
                for member in members {
                    let field = Self::new(&member.datatype, offsets)?;
                    if member
                        .byte_offset
                        .checked_add(field.source_size.to_u64())
                        .is_none_or(|end| end > u64::from(*size))
                    {
                        return Err(FormatError::CompoundFieldOutOfBounds {
                            name: member.name.clone(),
                            offset: member.byte_offset,
                            field_size: member.datatype.type_size(),
                            compound_size: *size,
                        });
                    }
                    if field.has_references() {
                        fields.push((member.byte_offset.to_usize()?, field));
                    }
                }
                let size = size.to_usize()?;
                (size, size, ElementKind::Compound(fields))
            }
            _ => {
                let size = datatype.type_size().to_usize()?;
                (size, size, ElementKind::Fixed)
            }
        };
        Ok(Self {
            source_size,
            target_size,
            kind,
        })
    }

    /// Rewrites one canonical element at `source` into `target` and adjusts staged patch positions.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if patch metadata disagrees with a reference.
    /// Propagates buffer-range, defined-address, and host-size conversion errors.
    fn write(
        &self,
        raw: &mut [u8],
        source: usize,
        target: usize,
        offsets: OffsetWidth,
        patches: &mut Option<&mut [usize]>,
        visited: &mut usize,
    ) -> Result<(), FormatError> {
        match &self.kind {
            ElementKind::Reference => {
                bytes::ensure_len(raw, source, REFERENCE_BYTES)?;
                let count = LittleEndian::read_u32(&raw[source..source + SEQUENCE_LENGTH_BYTES]);
                let address = bytes::read_offset_width(
                    raw,
                    source + SEQUENCE_LENGTH_BYTES,
                    OffsetWidth::Eight,
                )?;
                let index = LittleEndian::read_u32(
                    &raw[source + REFERENCE_BYTES - OBJECT_INDEX_BYTES..source + REFERENCE_BYTES],
                );
                let address = if let Some(patches) = patches {
                    if address == 0 {
                        if count != 0 || index != 0 {
                            return Err(FormatError::Internal(
                                "a heap-backed variable-length reference has no staged patch"
                                    .into(),
                            ));
                        }
                    } else {
                        let ordinal = (address - 1).to_usize()?;
                        let patch = patches.get_mut(ordinal).ok_or_else(|| {
                            FormatError::Internal(
                                "a variable-length patch ordinal is absent".into(),
                            )
                        })?;
                        if *patch != source {
                            return Err(FormatError::Internal(
                                "a variable-length patch refers to another element".into(),
                            ));
                        }
                        *patch = target;
                        *visited += 1;
                    }
                    0
                } else {
                    address
                };
                bytes::ensure_len(raw, target, self.target_size)?;
                raw[target..target + SEQUENCE_LENGTH_BYTES].copy_from_slice(&count.to_le_bytes());
                bytes::write_defined_stored_address_at(
                    raw,
                    target + SEQUENCE_LENGTH_BYTES,
                    StoredAddress::new(address),
                    offsets,
                )?;
                let index_at = target + self.target_size - OBJECT_INDEX_BYTES;
                raw[index_at..index_at + OBJECT_INDEX_BYTES].copy_from_slice(&index.to_le_bytes());
            }
            ElementKind::Array { count, base } => {
                for element in 0..*count {
                    base.write(
                        raw,
                        source + element * base.source_size,
                        target + element * base.target_size,
                        offsets,
                        patches,
                        visited,
                    )?;
                }
            }
            ElementKind::Compound(fields) => {
                for (offset, field) in fields {
                    field.write(
                        raw,
                        source + offset,
                        target + offset,
                        offsets,
                        patches,
                        visited,
                    )?;
                }
            }
            ElementKind::Fixed => {}
        }
        if source == target && self.source_size > self.target_size {
            bytes::ensure_len(raw, source, self.source_size)?;
            raw[target + self.target_size..source + self.source_size].fill(0);
        }
        Ok(())
    }

    fn has_references(&self) -> bool {
        match &self.kind {
            ElementKind::Fixed => false,
            ElementKind::Compound(fields) => !fields.is_empty(),
            _ => true,
        }
    }
}

enum ElementKind {
    Array {
        count: usize,
        base: Box<ElementLayout>,
    },
    Compound(Vec<(usize, ElementLayout)>),
    Fixed,
    Reference,
}

/// The four-byte count written before the heap ID by `H5T__vlen_disk_write`
/// (`H5Tvlen.c`, HDF5 2.2.0).
const SEQUENCE_LENGTH_BYTES: usize = 4;
/// The four-byte object index in "Global Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
const OBJECT_INDEX_BYTES: usize = 4;
/// The canonical reference length: [`SEQUENCE_LENGTH_BYTES`], an eight-byte heap address, and
/// [`OBJECT_INDEX_BYTES`].
const REFERENCE_BYTES: usize = SEQUENCE_LENGTH_BYTES + 8 + OBJECT_INDEX_BYTES;
/// The four-byte Size field in "The Datatype Message" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DATATYPE_SIZE_BYTES: u8 = 4;

#[cfg(test)]
mod tests {
    use alloc::vec;

    use hdf5_pure_core::__private;
    use hdf5_pure_core::CharacterSet;
    use hdf5_pure_core::StringPadding;
    use hdf5_pure_core::VariableLengthType;
    use rstest::rstest;
    use test_util::global_heap;
    use test_util::widths::Widths;

    use super::*;

    fn string_type() -> Datatype {
        Datatype::VariableLength(VariableLengthType::string(
            StringPadding::NullTerminate,
            CharacterSet::Utf8,
        ))
    }

    #[rstest]
    #[case::two(OffsetWidth::Two, 10)]
    #[case::four(OffsetWidth::Four, 12)]
    #[case::eight(OffsetWidth::Eight, 16)]
    fn narrowing_preserves_compound_fields_and_patch_order(
        #[case] offsets: OffsetWidth,
        #[case] reference_size: usize,
    ) {
        let datatype = Datatype::Compound {
            size: 64,
            members: vec![__private::compound_member(
                "strings".into(),
                8,
                Datatype::Array {
                    base_type: Box::new(string_type()),
                    dimensions: vec![2],
                },
            )],
        };
        let mut raw = vec![0xcc; 64];
        raw[8..24].copy_from_slice(&global_heap::reference(2, 0, 2, Widths::EIGHT));
        raw[24..40].copy_from_slice(&global_heap::reference(3, 0, 1, Widths::EIGHT));
        let capacity = raw.capacity();
        let mut patches = [24, 8];
        narrow_variable_length_references(offsets, &datatype, &mut raw, Some(&mut patches))
            .unwrap();
        assert_eq!(raw.len(), 64);
        assert_eq!(raw.capacity(), capacity);
        assert_eq!(patches, [8 + reference_size, 8]);
        assert_eq!(&raw[..8], &[0xcc; 8]);
        assert_eq!(&raw[40..], &[0xcc; 24]);
        let mut first = vec![0; reference_size];
        first[..4].copy_from_slice(&2u32.to_le_bytes());
        first[reference_size - 4..].copy_from_slice(&2u32.to_le_bytes());
        let mut second = vec![0; reference_size];
        second[..4].copy_from_slice(&3u32.to_le_bytes());
        second[reference_size - 4..].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(&raw[8..8 + reference_size], first);
        assert_eq!(&raw[8 + reference_size..8 + 2 * reference_size], second);
        assert!(
            raw[8 + 2 * reference_size..40]
                .iter()
                .all(|byte| *byte == 0)
        );
    }

    #[rstest]
    #[case::exact(65_534, Ok(()))]
    #[case::undefined(65_535, Err(FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 }))]
    #[case::beyond(65_536, Err(FormatError::AddressTooLarge { address: 65_536, offset_size: 2 }))]
    fn narrowing_checks_defined_heap_addresses(
        #[case] address: u64,
        #[case] expected: Result<(), FormatError>,
    ) {
        let mut raw = global_heap::reference(1, address, 1, Widths::EIGHT);
        assert_eq!(
            narrow_variable_length_references(OffsetWidth::Two, &string_type(), &mut raw, None),
            expected
        );
        if expected.is_ok() {
            assert_eq!(raw, [1, 0, 0, 0, 0xfe, 0xff, 1, 0, 0, 0]);
        }
    }

    #[test]
    fn narrowing_compacts_reference_arrays_in_place() {
        let datatype = Datatype::Array {
            base_type: Box::new(string_type()),
            dimensions: vec![3],
        };
        let mut raw = [
            global_heap::reference(1, 0, 1, Widths::EIGHT),
            global_heap::reference(0, 0, 0, Widths::EIGHT),
            global_heap::reference(2, 0, 2, Widths::EIGHT),
        ]
        .concat();
        let capacity = raw.capacity();
        let mut patches = [32, 0];
        narrow_variable_length_references(
            OffsetWidth::Two,
            &datatype,
            &mut raw,
            Some(&mut patches),
        )
        .unwrap();
        assert_eq!(raw.len(), 30);
        assert_eq!(raw.capacity(), capacity);
        assert_eq!(patches, [20, 0]);
        assert_eq!(
            raw,
            [
                1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0,
                0, 0
            ]
        );
    }
}
