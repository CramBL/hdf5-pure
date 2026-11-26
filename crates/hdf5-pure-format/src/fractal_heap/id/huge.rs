use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use super::frame::FractalHeapIdType;
use crate::address::StoredAddress;
use crate::bytes;
use crate::error::FormatError;
use crate::fractal_heap::location::{self, FilteredHugeObjectLocation, HugeObjectLocationEncoding};
use crate::fractal_heap::{FractalHeapFiltering, FractalHeapHeader};
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

/// The location of a huge object, as its heap ID gives it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HugeObjectReference {
    /// The heap ID holds this key, which the huge-object B-tree of the heap maps to the address
    /// and the length of the object.
    Indexed(HugeIndirectKey),
    /// The heap ID holds a key in the heap's type-2 filtered huge-object tracker.
    FilteredIndexed(HugeIndirectKey),
    /// The heap ID holds the address and the length of the object.
    Inline {
        /// The address of the object.
        addr: StoredAddress,
        /// The length of the object in bytes.
        len: u64,
    },
    /// The heap ID holds the filtered object's location, sizes, and filter mask.
    FilteredInline(FilteredHugeObjectLocation),
}

/// The encoded width of a numeric key in an indirect huge-object heap ID.
///
/// The tracker record's key uses the file's length width in both dialects. Callers compare
/// decoded numeric values when the heap ID and tracker record use different widths.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HugeIndirectKeyEncoding {
    /// Uses the smaller of the heap ID payload length and eight bytes.
    ///
    /// `H5HF__huge_init` applies this rule in `H5HFhuge.c` (HDF5 2.2.0).
    Libhdf5,
    /// Uses the file's length width.
    ///
    /// The key is defined in "Fractal Heap" of the [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
    PublishedSpec,
}

/// A numeric huge-object tracker key decoded under an explicit width dialect.
///
/// The value fits the tracker's length field. When parsing an indirect huge-object ID,
/// [`FractalHeapIdLayout::parse`](super::FractalHeapIdLayout::parse) decodes this key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HugeIndirectKey {
    value: u64,
    encoding: HugeIndirectKeyEncoding,
}

impl HugeIndirectKey {
    /// Returns the numeric key for comparison with the tracker record's object ID.
    pub const fn value(self) -> u64 {
        self.value
    }
    /// Returns the width dialect used to decode the heap ID's key.
    pub const fn encoding(self) -> HugeIndirectKeyEncoding {
        self.encoding
    }
}

impl FractalHeapHeader {
    /// Returns the heap ID of the huge object whose key in the huge-object B-tree is `huge_id`.
    ///
    /// An unfiltered heap uses indirect IDs when an address of `offset_width` and a length of
    /// `length_width` exceed the space after the prefix.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFilteredHeapObject`] if the heap filters its objects,
    /// and [`FormatError::Internal`] if the huge IDs of the heap store an address and a length, or
    /// if `huge_id` does not fit the ID.
    pub fn encode_huge_id(
        &self,
        huge_id: u64,
        offset_width: OffsetWidth,
        length_width: LengthWidth,
    ) -> Result<Vec<u8>, FormatError> {
        if self.filtering.is_filtered() {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        if self.huge_ids_direct_with_widths(FormatWidths {
            offsets: offset_width,
            lengths: length_width,
        }) {
            return Err(FormatError::Internal(
                "a key was encoded into a heap ID that holds its object's address".into(),
            ));
        }
        self.encode_id(FractalHeapIdType::Huge, Some(huge_id), "huge object key")
    }

    /// Encodes an unfiltered huge object's address and length directly into its heap ID.
    ///
    /// Direct IDs are defined in "Fractal Heap" of the [format specification, version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFilteredHeapObject`] for a filtered heap, or
    /// [`FormatError::Internal`] if the ID cannot hold a direct address and length. Propagates
    /// defined-address and length errors from [`HugeObjectLocationEncoding::new`].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
    pub(crate) fn encode_huge_direct_id(
        &self,
        address: StoredAddress,
        length: u64,
        widths: FormatWidths,
    ) -> Result<Vec<u8>, FormatError> {
        if self.filtering.is_filtered() {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        if !self.huge_ids_direct_with_widths(widths) {
            return Err(FormatError::Internal(
                "an address was encoded into a heap ID that holds a B-tree key".into(),
            ));
        }
        let mut id = Vec::with_capacity(usize::from(self.heap_id_length));
        id.push(FractalHeapIdType::Huge.first_byte());
        HugeObjectLocationEncoding::new(widths, address, length)?.append(&mut id);
        id.resize(usize::from(self.heap_id_length), 0);
        Ok(id)
    }

    /// Returns a heap ID of `id_type` whose bytes after the first store `payload` little-endian.
    ///
    /// A `payload` of `None` is one too wide for the ID, and the error names it as `what`.
    fn encode_id(
        &self,
        id_type: FractalHeapIdType,
        payload: Option<u64>,
        what: &str,
    ) -> Result<Vec<u8>, FormatError> {
        let mut id = vec![0; usize::from(self.heap_id_length)];
        let Some((first, rest)) = id.split_first_mut() else {
            return Err(FormatError::Internal(
                "a heap ID was encoded for a heap whose IDs are empty".into(),
            ));
        };
        let too_wide = || {
            FormatError::Internal(format!(
                "a {what} does not fit a {}-byte heap ID",
                self.heap_id_length
            ))
        };
        let payload = payload.ok_or_else(too_wide)?.to_le_bytes();
        let (low, high) = payload.split_at(rest.len().min(payload.len()));
        if high.iter().any(|&byte| byte != 0) {
            return Err(too_wide());
        }
        *first = id_type.first_byte();
        for (byte, &value) in rest.iter_mut().zip(low) {
            *byte = value;
        }
        Ok(id)
    }

    /// Returns `true` if a huge object ID has room for the object's location after its prefix.
    ///
    /// An unfiltered location requires an address and a length. A filtered location also
    /// includes a four-byte filter mask and the decoded length. Returns `false` if the
    /// location does not fit or either field width is unsupported.
    pub fn huge_ids_direct(&self, offset_size: u8, length_size: u8) -> bool {
        let Ok(widths) = FormatWidths::from_sizes(offset_size, length_size) else {
            return false;
        };
        self.huge_ids_direct_with_widths(widths)
    }

    /// Returns whether huge heap IDs store object addresses and lengths at `widths`.
    pub fn huge_ids_direct_with_widths(&self, widths: FormatWidths) -> bool {
        if self.filtering.is_filtered() {
            let avail = usize::from(self.heap_id_length).saturating_sub(1);
            let offset_size = usize::from(widths.offsets.get());
            let length_size = usize::from(widths.lengths.get());
            avail >= offset_size + length_size + 4 + length_size
        } else {
            Self::unfiltered_huge_ids_direct(self.heap_id_length, widths)
        }
    }

    /// Returns `true` if an unfiltered heap ID can hold its prefix, address, and length.
    ///
    /// The fields are defined in "Fractal Heap" of the [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
    pub(crate) fn unfiltered_huge_ids_direct(heap_id_length: u16, widths: FormatWidths) -> bool {
        usize::from(heap_id_length).saturating_sub(HEAP_ID_PREFIX_BYTES)
            >= usize::from(widths.offsets.get()) + usize::from(widths.lengths.get())
    }
}

/// The fields of a huge heap ID for one heap, file width pair, and indirect key encoding.
#[derive(Clone, Copy, Debug)]
pub(super) struct HugeIdLayout {
    direct: bool,
    offsets: OffsetWidth,
    lengths: LengthWidth,
    tracker: StoredAddress,
    filter_count: Option<usize>,
    encoding: HugeIndirectKeyEncoding,
    key_width: usize,
}

impl HugeIdLayout {
    pub(super) fn new(
        header: &FractalHeapHeader,
        offsets: OffsetWidth,
        lengths: LengthWidth,
        encoding: HugeIndirectKeyEncoding,
    ) -> Self {
        Self {
            direct: header.huge_ids_direct_with_widths(FormatWidths { offsets, lengths }),
            offsets,
            lengths,
            tracker: header.btree_huge_objects_address,
            filter_count: match &header.filtering {
                FractalHeapFiltering::Unfiltered => None,
                FractalHeapFiltering::Filtered { pipeline, .. } => Some(pipeline.filters().len()),
            },
            encoding,
            key_width: match encoding {
                HugeIndirectKeyEncoding::PublishedSpec => usize::from(lengths.get()),
                // H5HF__huge_init in H5HFhuge.c, HDF5 2.2.0, bounds the indirect key by
                // the ID payload length and the eight-byte native huge ID.
                HugeIndirectKeyEncoding::Libhdf5 => {
                    usize::from(header.heap_id_length).saturating_sub(1).min(8)
                }
            },
        }
    }

    pub(super) fn parse(self, payload: &[u8]) -> Result<HugeObjectReference, FormatError> {
        let HugeIdLayout {
            direct,
            offsets,
            lengths,
            tracker,
            filter_count,
            encoding,
            key_width,
        } = self;
        if tracker.get() >= StoredAddress::undefined(offsets).get() {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        Ok(if direct {
            if let Some(filter_count) = filter_count {
                HugeObjectReference::FilteredInline(FilteredHugeObjectLocation::parse(
                    payload,
                    offsets,
                    lengths,
                    filter_count,
                )?)
            } else {
                let (addr, len) = location::read_unfiltered(
                    &mut bytes::ByteCursor::new(payload),
                    FormatWidths { offsets, lengths },
                )?;
                HugeObjectReference::Inline {
                    addr,
                    len: len.get(),
                }
            }
        } else {
            if key_width == 0 || key_width > payload.len() {
                return Err(FormatError::InvalidFractalHeapGeometry);
            }
            let key = bytes::ByteCursor::new(payload).uint(key_width)?;
            if !crate::width::length_width_holds(lengths, key) {
                return Err(FormatError::ValueTooLargeForRepresentation {
                    value: key,
                    target: "huge-object tracker key",
                });
            }
            let key = HugeIndirectKey {
                value: key,
                encoding,
            };
            if filter_count.is_some() {
                HugeObjectReference::FilteredIndexed(key)
            } else {
                HugeObjectReference::Indexed(key)
            }
        })
    }
}

/// The version-and-type byte of a heap ID, defined in "Fractal Heap" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
const HEAP_ID_PREFIX_BYTES: usize = 1;

#[cfg(test)]
#[path = "tests/huge.rs"]
mod tests;
