use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::btree::v2::BTreeV2Header;
use crate::btree::v2::BTreeV2Record;
use crate::bytes;
use crate::error::FormatError;
use crate::fractal_heap::location::{self, HugeObjectLocationEncoding};
use crate::fractal_heap::{FilteredHugeObjectLocation, FractalHeapFiltering, FractalHeapHeader};
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

impl BTreeV2Record {
    /// Parses the record as a record of the huge-object index of a fractal heap, type 1.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the record ends before its three fields.
    pub fn huge_object(
        &self,
        offset_width: OffsetWidth,
        length_width: LengthWidth,
    ) -> Result<HugeObjectRecord, FormatError> {
        let os = usize::from(offset_width.get());
        let ls = usize::from(length_width.get());
        Ok(HugeObjectRecord {
            address: StoredAddress::new(bytes::read_offset_width(&self.data, 0, offset_width)?),
            length: bytes::read_length_width(&self.data, os, length_width)?,
            id: bytes::read_length_width(&self.data, os + ls, length_width)?,
        })
    }

    /// Parses the record as a record of the huge-object index of a fractal heap, type 3.
    ///
    /// Type 3 is the non-filtered layout used when heap IDs hold the huge object's address and
    /// length directly. The B-tree still tracks every huge object so the heap can enumerate and
    /// delete all of them.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the record ends before its two fields.
    pub fn huge_object_direct(
        &self,
        offset_width: OffsetWidth,
        length_width: LengthWidth,
    ) -> Result<HugeObjectDirectRecord, FormatError> {
        let os = usize::from(offset_width.get());
        Ok(HugeObjectDirectRecord {
            address: StoredAddress::new(bytes::read_offset_width(&self.data, 0, offset_width)?),
            length: bytes::read_length_width(&self.data, os, length_width)?,
        })
    }
}

/// A record of the version 2 B-tree that indexes the huge objects of a fractal heap, type 1: the
/// objects the heap does not filter and whose heap IDs store the key of their record.
///
/// The record is defined in "Version 2 B-trees" of the [format specification, version
/// 4.0][spec], which calls the key the Huge Object ID.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HugeObjectRecord {
    /// The address of the object.
    pub address: StoredAddress,
    /// The length of the object in bytes.
    pub length: u64,
    /// The key of the object, which its heap ID stores.
    pub id: u64,
}

/// A record of a fractal heap huge-object B-tree, type 3: the object address and length.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HugeObjectDirectRecord {
    /// The address of the object.
    pub address: StoredAddress,
    /// The length of the object in bytes.
    pub length: u64,
}

impl HugeObjectDirectRecord {
    /// Returns the encoded length of a type-3 huge-object record.
    pub const fn size(offset_width: OffsetWidth, length_width: LengthWidth) -> u16 {
        offset_width.get() as u16 + length_width.get() as u16
    }

    /// Appends the address and length of an unfiltered, directly accessed huge object.
    ///
    /// The record is type 3 in "Version 2 B-trees" of the [format specification, version
    /// 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if the address exceeds `offsets`,
    /// [`FormatError::DefinedAddressIsUndefined`] if it equals that width's undefined marker, or
    /// [`FormatError::LengthTooLarge`] if the length exceeds `lengths`. Leaves `buf` unchanged on
    /// error.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
    pub(crate) fn encode(
        &self,
        buf: &mut Vec<u8>,
        offsets: OffsetWidth,
        lengths: LengthWidth,
    ) -> Result<(), FormatError> {
        HugeObjectLocationEncoding::new(
            FormatWidths { offsets, lengths },
            self.address,
            self.length,
        )?
        .append(buf);
        Ok(())
    }
}

impl HugeObjectRecord {
    /// Returns the length in bytes of a record whose address is `offset_width` bytes wide and
    /// whose length and key are `length_width` bytes wide each.
    pub const fn size(offset_width: OffsetWidth, length_width: LengthWidth) -> u16 {
        offset_width.get() as u16 + 2 * length_width.get() as u16
    }

    /// Appends one complete record after checking the address, length, and key.
    ///
    /// Uses `offset_width` bytes for the address and `length_width` bytes each for the length and
    /// key. Leaves `buf` unchanged if a field cannot be represented.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if the address does not fit `offset_width`,
    /// [`FormatError::DefinedAddressIsUndefined`] if it equals that width's undefined address,
    /// and [`FormatError::LengthTooLarge`] if the length or key does not fit `length_width`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::{FormatError, LengthWidth, OffsetWidth, StoredAddress};
    /// use hdf5_pure_format::__private::HugeObjectRecord;
    ///
    /// let record = HugeObjectRecord {
    ///     address: StoredAddress::new(1024),
    ///     length: 17,
    ///     id: 65_536,
    /// };
    /// let mut bytes = vec![0xAB, 0xCD];
    /// let error = record.encode(&mut bytes, OffsetWidth::Two, LengthWidth::Two).unwrap_err();
    /// assert_eq!(error, FormatError::LengthTooLarge { length: 65_536, length_size: 2 });
    /// assert_eq!(bytes, [0xAB, 0xCD]);
    /// ```
    pub fn encode(
        &self,
        buf: &mut Vec<u8>,
        offset_width: OffsetWidth,
        length_width: LengthWidth,
    ) -> Result<(), FormatError> {
        let Self {
            address,
            length,
            id,
        } = *self;
        let location = HugeObjectLocationEncoding::new(
            FormatWidths {
                offsets: offset_width,
                lengths: length_width,
            },
            address,
            length,
        )?;
        bytes::require_length(id, length_width)?;
        location.append(buf);
        bytes::write_length(buf, id, length_width);
        Ok(())
    }
}

/// A type-2 tracker record for an indirectly accessed, filtered huge object.
///
/// The record stores the encoded object's address and length, its filter mask and decoded size,
/// and its numeric ID, in that order. Obtained from [`HugeObjectRecordDecoder::parse`].
///
/// The record is defined in "Version 2 B-trees" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FilteredHugeObjectRecord {
    location: FilteredHugeObjectLocation,
    id: u64,
}

impl FilteredHugeObjectRecord {
    /// Returns the record length: one address, three lengths, and a four-byte filter mask.
    pub const fn size(offsets: OffsetWidth, lengths: LengthWidth) -> u16 {
        offsets.get() as u16 + 3 * lengths.get() as u16 + HUGE_OBJECT_FILTER_MASK_LEN
    }

    fn parse(
        data: &[u8],
        offsets: OffsetWidth,
        lengths: LengthWidth,
        filter_count: usize,
    ) -> Result<Self, FormatError> {
        let required = usize::from(Self::size(offsets, lengths));
        if data.len() != required {
            return Err(FormatError::UnexpectedHugeObjectBTree {
                tree_type: BTREE_V2_FILTERED_HUGE_OBJECT,
                record_size: data.len(),
                required,
            });
        }
        Ok(Self {
            location: FilteredHugeObjectLocation::parse(data, offsets, lengths, filter_count)?,
            id: bytes::read_length_width(data, required - usize::from(lengths.get()), lengths)?,
        })
    }

    /// Returns the encoded payload's location and filter metadata.
    pub const fn location(self) -> FilteredHugeObjectLocation {
        self.location
    }
    /// Returns the numeric key for lookup from an indirect huge-object heap ID.
    pub const fn id(self) -> u64 {
        self.id
    }
}

/// A type-4 tracker record for a directly accessed, filtered huge object.
///
/// The record stores the encoded object's address and length, its filter mask, and its decoded
/// size, in that order. Obtained from [`HugeObjectRecordDecoder::parse`].
///
/// The record is defined in "Version 2 B-trees" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FilteredHugeObjectDirectRecord {
    location: FilteredHugeObjectLocation,
}

impl FilteredHugeObjectDirectRecord {
    /// Returns the record length: one address, two lengths, and a four-byte filter mask.
    pub const fn size(offsets: OffsetWidth, lengths: LengthWidth) -> u16 {
        offsets.get() as u16 + 2 * lengths.get() as u16 + HUGE_OBJECT_FILTER_MASK_LEN
    }

    fn parse(
        data: &[u8],
        offsets: OffsetWidth,
        lengths: LengthWidth,
        filter_count: usize,
    ) -> Result<Self, FormatError> {
        let required = usize::from(Self::size(offsets, lengths));
        if data.len() != required {
            return Err(FormatError::UnexpectedHugeObjectBTree {
                tree_type: BTREE_V2_FILTERED_HUGE_OBJECT_DIRECT,
                record_size: data.len(),
                required,
            });
        }
        Ok(Self {
            location: FilteredHugeObjectLocation::parse(data, offsets, lengths, filter_count)?,
        })
    }

    /// Returns the encoded payload's location and filter metadata.
    pub const fn location(self) -> FilteredHugeObjectLocation {
        self.location
    }
}

/// The decoded fields of one huge-object tracker record, selected from the heap's parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HugeObjectTrackerRecord {
    /// A type-1 record for an unfiltered object located through a numeric key.
    Indirect(HugeObjectRecord),
    /// A type-2 record for a filtered object located through a numeric key.
    FilteredIndirect(FilteredHugeObjectRecord),
    /// A type-3 record for an unfiltered object whose location is stored in its heap ID.
    Direct(HugeObjectDirectRecord),
    /// A type-4 record for a filtered object whose location is stored in its heap ID.
    FilteredDirect(FilteredHugeObjectDirectRecord),
}

/// The huge-object tracker encoding required by a fractal heap.
///
/// A reader derives the record type and exact size from the heap's filter state and ID capacity.
/// [`resolve`](Self::resolve) checks the tracker header before the caller decodes record bytes
/// with [`HugeObjectRecordDecoder::parse`]. The caller reads the metadata and traverses the tree.
///
/// # Examples
///
/// ```
/// # use hdf5_pure_core::StoredAddress;
/// # use hdf5_pure_core::__private::StoredAddressExt;
/// # use hdf5_pure_format::__private::{
/// #     BTreeV2Header, FilterPipeline, FILTER_PIPELINE_VERSION_TWO, FormatWidths,
/// #     FractalHeapFiltering, FractalHeapHeader, HugeObjectRecordContext,
/// #     HugeObjectTrackerRecord, LengthWidth, OffsetWidth,
/// # };
/// # use hdf5_pure_format::FormatError;
/// # use test_util::{fractal_heap, widths::Widths};
/// # let mut heap = FractalHeapHeader::parse_bytes(
/// #     &fractal_heap::Header::new(256).build(Widths::EIGHT),
/// #     OffsetWidth::Eight,
/// #     LengthWidth::Eight,
/// # )?;
/// # heap.btree_huge_objects_address = StoredAddress::new(512);
/// # heap.filtering = FractalHeapFiltering::Filtered {
/// #     root_direct_block_size: 0,
/// #     root_filter_mask: 0,
/// #     pipeline: FilterPipeline::new(FILTER_PIPELINE_VERSION_TWO, Vec::new())?,
/// # };
/// # let widths = FormatWidths { offsets: OffsetWidth::Four, lengths: LengthWidth::Two };
/// # let tracker = BTreeV2Header {
/// #     tree_type: hdf5_pure_format::__private::BTREE_V2_FILTERED_HUGE_OBJECT,
/// #     node_size: 512, record_size: 14, depth: 0,
/// #     root_node_address: StoredAddress::new(1024), num_records_in_root: 1, total_records: 1,
/// # };
/// let context = HugeObjectRecordContext::new(&heap, widths)?;
/// let decoder = context.resolve(&tracker)?;
/// let encoded = [0, 16, 0, 0, 17, 0, 0, 0, 0, 0, 17, 0, 5, 0];
/// let HugeObjectTrackerRecord::FilteredIndirect(record) = decoder.parse(&encoded)? else {
///     unreachable!();
/// };
/// assert_eq!(record.id(), 5);
/// assert_eq!(record.location().address(), StoredAddress::new(4096));
/// assert_eq!(record.location().encoded_size(), 17);
/// assert_eq!(record.location().decoded_size(), 17);
/// # Ok::<(), FormatError>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HugeObjectRecordContext {
    widths: FormatWidths,
    encoding: HugeObjectRecordEncoding,
}

impl HugeObjectRecordContext {
    /// Selects a tracker encoding from the heap's filter state and ID capacity at `widths`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] if the tracker address is undefined
    /// or exceeds the address width.
    pub fn new(header: &FractalHeapHeader, widths: FormatWidths) -> Result<Self, FormatError> {
        if header.btree_huge_objects_address.get() >= StoredAddress::undefined(widths.offsets).get()
        {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        let direct = header.huge_ids_direct_with_widths(widths);
        let encoding = match &header.filtering {
            FractalHeapFiltering::Unfiltered => {
                if direct {
                    HugeObjectRecordEncoding::Direct
                } else {
                    HugeObjectRecordEncoding::Indirect
                }
            }
            FractalHeapFiltering::Filtered { pipeline, .. } => {
                let filter_count = pipeline.filters().len();
                if direct {
                    HugeObjectRecordEncoding::FilteredDirect { filter_count }
                } else {
                    HugeObjectRecordEncoding::FilteredIndirect { filter_count }
                }
            }
        };
        Ok(Self { widths, encoding })
    }

    /// Returns the required tracker type, from 1 through 4.
    pub const fn tree_type(self) -> u8 {
        match self.encoding {
            HugeObjectRecordEncoding::Indirect => BTREE_V2_HUGE_OBJECT,
            HugeObjectRecordEncoding::FilteredIndirect { .. } => BTREE_V2_FILTERED_HUGE_OBJECT,
            HugeObjectRecordEncoding::Direct => BTREE_V2_HUGE_OBJECT_DIRECT,
            HugeObjectRecordEncoding::FilteredDirect { .. } => BTREE_V2_FILTERED_HUGE_OBJECT_DIRECT,
        }
    }

    /// Returns the exact encoded length of each tracker record in bytes.
    pub const fn record_size(self) -> u16 {
        let FormatWidths { offsets, lengths } = self.widths;
        match self.encoding {
            HugeObjectRecordEncoding::Indirect => HugeObjectRecord::size(offsets, lengths),
            HugeObjectRecordEncoding::FilteredIndirect { .. } => {
                FilteredHugeObjectRecord::size(offsets, lengths)
            }
            HugeObjectRecordEncoding::Direct => HugeObjectDirectRecord::size(offsets, lengths),
            HugeObjectRecordEncoding::FilteredDirect { .. } => {
                FilteredHugeObjectDirectRecord::size(offsets, lengths)
            }
        }
    }

    /// Checks the tracker header's type and exact record size before decoding its records.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedHugeObjectBTree`] if the type or size differs from this
    /// context's encoding.
    pub fn resolve(self, header: &BTreeV2Header) -> Result<HugeObjectRecordDecoder, FormatError> {
        if header.tree_type != self.tree_type() || header.record_size != self.record_size() {
            return Err(FormatError::UnexpectedHugeObjectBTree {
                tree_type: header.tree_type,
                record_size: usize::from(header.record_size),
                required: usize::from(self.record_size()),
            });
        }
        Ok(HugeObjectRecordDecoder { layout: self })
    }
}

/// A huge-object tracker record decoder whose header type and record size have been checked.
///
/// Obtained from [`HugeObjectRecordContext::resolve`]. The caller supplies individual record
/// bytes from the tracker for [`parse`](Self::parse).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HugeObjectRecordDecoder {
    layout: HugeObjectRecordContext,
}

impl HugeObjectRecordDecoder {
    /// Decodes exactly one record using the checked tracker encoding.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedHugeObjectBTree`] if `data` has a different length from
    /// the expected record size. Returns [`FormatError::InvalidFractalHeapGeometry`] if the
    /// address is undefined, a required size is zero, the mask contains bits outside the
    /// pipeline, or the sizes differ with all filters skipped. Returns
    /// [`FormatError::OffsetOverflow`] if the encoded object's range end exceeds [`u64::MAX`].
    pub fn parse(self, data: &[u8]) -> Result<HugeObjectTrackerRecord, FormatError> {
        let Self { layout } = self;
        if data.len() != usize::from(layout.record_size()) {
            return Err(FormatError::UnexpectedHugeObjectBTree {
                tree_type: layout.tree_type(),
                record_size: data.len(),
                required: usize::from(layout.record_size()),
            });
        }
        let FormatWidths { offsets, lengths } = layout.widths;
        Ok(match layout.encoding {
            HugeObjectRecordEncoding::FilteredIndirect { filter_count } => {
                HugeObjectTrackerRecord::FilteredIndirect(FilteredHugeObjectRecord::parse(
                    data,
                    offsets,
                    lengths,
                    filter_count,
                )?)
            }
            HugeObjectRecordEncoding::FilteredDirect { filter_count } => {
                HugeObjectTrackerRecord::FilteredDirect(FilteredHugeObjectDirectRecord::parse(
                    data,
                    offsets,
                    lengths,
                    filter_count,
                )?)
            }
            HugeObjectRecordEncoding::Indirect | HugeObjectRecordEncoding::Direct => {
                let mut cursor = bytes::ByteCursor::new(data);
                let (address, length) = location::read_unfiltered(&mut cursor, layout.widths)?;
                if layout.encoding == HugeObjectRecordEncoding::Indirect {
                    HugeObjectTrackerRecord::Indirect(HugeObjectRecord {
                        address,
                        length: length.get(),
                        id: cursor.length(lengths)?,
                    })
                } else {
                    HugeObjectTrackerRecord::Direct(HugeObjectDirectRecord {
                        address,
                        length: length.get(),
                    })
                }
            }
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HugeObjectRecordEncoding {
    Indirect,
    FilteredIndirect { filter_count: usize },
    Direct,
    FilteredDirect { filter_count: usize },
}

/// The type of a version 2 B-tree that indexes the huge objects of a fractal heap that the heap
/// does not filter and whose heap IDs store a key.
///
/// The type is defined in the Type table of "Version 2 B-trees" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub const BTREE_V2_HUGE_OBJECT: u8 = 1;

/// The type of a version 2 B-tree that indexes filtered huge objects accessed by numeric key.
///
/// The type is defined in the same table as [`BTREE_V2_HUGE_OBJECT`].
pub const BTREE_V2_FILTERED_HUGE_OBJECT: u8 = 2;

/// The type of a version 2 B-tree that indexes non-filtered huge objects whose heap IDs carry the
/// object address and length directly.
pub const BTREE_V2_HUGE_OBJECT_DIRECT: u8 = 3;

/// The type of a version 2 B-tree that tracks filtered huge objects located by their heap IDs.
///
/// The type is defined in the same table as [`BTREE_V2_HUGE_OBJECT`].
pub const BTREE_V2_FILTERED_HUGE_OBJECT_DIRECT: u8 = 4;

/// The filter-mask width in type-2 and type-4 records, from the same section as
/// [`BTREE_V2_HUGE_OBJECT`].
const HUGE_OBJECT_FILTER_MASK_LEN: u16 = 4;

#[cfg(test)]
#[path = "tests/huge.rs"]
mod tests;
