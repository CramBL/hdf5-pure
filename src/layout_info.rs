//! Public, curated introspection of a dataset's on-disk storage layout and
//! filter pipeline (issue #149).
//!
//! These types are decoded from the HDF5 data-layout and filter-pipeline
//! messages but deliberately omit on-disk encoding artifacts (message version
//! numbers, the addresses of chunk index structures, and a filtered single
//! chunk's stored size and filter mask), so the public surface is not welded to
//! the internal parse representation. Obtain them from the [`Dataset`] accessors
//! [`layout`], [`chunk_index`], [`chunks`], and [`filter_pipeline`].
//!
//! [`Dataset`]: crate::Dataset
//! [`layout`]: crate::Dataset::layout
//! [`chunk_index`]: crate::Dataset::chunk_index
//! [`chunks`]: crate::Dataset::chunks
//! [`filter_pipeline`]: crate::Dataset::filter_pipeline

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::fmt;
use core::num::NonZeroU32;

use hdf5_pure_filter::__private::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZF, FILTER_NBIT, FILTER_RESERVED,
    FILTER_SCALEOFFSET, FILTER_SHUFFLE, FILTER_SZIP, FILTER_ZFP,
};

use crate::DeflateLevel;
use crate::FormatError;
use crate::ScaleOffset;
use crate::data_layout::ChunkIndexLayout;
use crate::display::{Dims, EscapedName};
use crate::filter_pipeline::FilterDescription;

/// How a dataset's raw data is arranged on disk.
///
/// The curated analogue of HDF5's layout class (`H5Pget_layout`), enriched with
/// the per-class facts needed to locate or size the data without decoding it.
/// Obtain it with [`Dataset::layout`](crate::Dataset::layout).
///
/// Use it to choose a reading strategy: a [`Contiguous`](Layout::Contiguous)
/// dataset is a single seek-and-read, while a [`Chunked`](Layout::Chunked)
/// dataset is read (and, for an appendable index, grown) one chunk at a time —
/// enumerate its chunks with [`Dataset::chunks`](crate::Dataset::chunks).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Layout {
    /// Stored inline in the dataset's object header, as used for tiny datasets.
    /// The bytes are already resident once the header is read, so there is no
    /// separate file region to seek to; `size` is the inline byte count.
    Compact {
        /// The number of raw bytes stored inline.
        size: u64,
    },
    /// Stored as one contiguous run of bytes.
    Contiguous {
        /// Absolute file offset of the first byte, or `None` when storage has
        /// not been allocated yet (a fixed-shape dataset that was never
        /// written). In that case `size` is the extent that *would* be written.
        address: Option<u64>,
        /// The length of the run in bytes.
        size: u64,
    },
    /// Stored as a grid of independently located (and optionally filtered)
    /// chunks. Filtered datasets are always chunked.
    Chunked {
        /// The chunk edge lengths, one per dataset dimension, in the same order
        /// as [`shape`](crate::Dataset::shape). This is the value returned by
        /// [`chunk_shape`](crate::Dataset::chunk_shape); the on-disk
        /// element-size dimension is stripped.
        chunk_shape: Vec<u64>,
        /// The index that maps chunk coordinates to file addresses, which
        /// governs append eligibility (see [`ChunkIndex`]).
        index: ChunkIndex,
    },
    /// A virtual dataset whose data is mapped from other datasets. Only the
    /// classification is exposed; the source mappings are not decoded.
    Virtual,
}

/// The kind of index a chunked dataset uses to locate its chunks.
///
/// The curated, named form of HDF5's chunk-index type. The index kind is fixed
/// at dataset creation by the shape and its extensibility, and it determines
/// whether the dataset can be grown in place: see
/// [`supports_inplace_append`](ChunkIndex::supports_inplace_append).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ChunkIndex {
    /// A version-1 B-tree indexes the chunks — the classic layout, used for any
    /// rank and any number of unlimited dimensions in older files.
    BTreeV1,
    /// A single chunk holds the entire dataset; there is no separate index
    /// structure.
    SingleChunk,
    /// Chunk addresses are computed arithmetically from each chunk's position
    /// (a fixed dataspace with every chunk allocated); there is no separate
    /// index structure.
    Implicit,
    /// A fixed array indexes a fixed number of chunks (a non-extensible
    /// dataspace with more than one chunk).
    FixedArray,
    /// An extensible array indexes chunks along a single unlimited dimension.
    /// This is the index [`Dataset::append`](crate::Dataset::append) and
    /// [`Dataset::append_staged`](crate::Dataset::append_staged) grow
    /// in place.
    ExtensibleArray,
    /// A version-2 B-tree indexes the chunks (several unlimited dimensions). The reader can
    /// enumerate and read this index, but the editor does not create or mutate it, and deletion
    /// does not yet reclaim its index structure.
    BTreeV2,
}

impl ChunkIndex {
    pub(crate) fn from_layout(index: ChunkIndexLayout) -> Self {
        match index {
            ChunkIndexLayout::BTreeV1 { .. } => ChunkIndex::BTreeV1,
            ChunkIndexLayout::SingleChunk { .. } => ChunkIndex::SingleChunk,
            ChunkIndexLayout::Implicit { .. } => ChunkIndex::Implicit,
            ChunkIndexLayout::FixedArray { .. } => ChunkIndex::FixedArray,
            ChunkIndexLayout::ExtensibleArray { .. } => ChunkIndex::ExtensibleArray,
            ChunkIndexLayout::BTreeV2 { .. } => ChunkIndex::BTreeV2,
        }
    }

    /// Whether a dataset with this index kind can be grown in place with
    /// [`Dataset::append`](crate::Dataset::append) — true only for
    /// [`ExtensibleArray`](ChunkIndex::ExtensibleArray).
    ///
    /// This reflects the index *structure* alone; an actual append also requires
    /// the dataset's first maximum dimension to be unlimited (see
    /// [`Dataset::maxshape`](crate::Dataset::maxshape)).
    #[must_use]
    pub const fn supports_inplace_append(self) -> bool {
        matches!(self, ChunkIndex::ExtensibleArray)
    }
}

/// The location and on-disk footprint of one stored chunk.
///
/// A `Chunk` is a lightweight record: enumerating chunks reads only the chunk
/// index, never the chunk data. To read one chunk, seek to
/// [`address`](Self::address), read exactly [`storage_size`](Self::storage_size)
/// bytes, then invert the dataset's
/// [`filter_pipeline`](crate::Dataset::filter_pipeline) in *reverse* order
/// (skipping the filters marked in [`filter_mask`](Self::filter_mask)). The
/// curated analogue of `H5Dget_chunk_info`; obtain these from
/// [`Dataset::chunks`](crate::Dataset::chunks).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Chunk {
    /// The logical offset of this chunk's first element within the dataset, one
    /// coordinate per dataset dimension (row-major, in elements). The origin
    /// chunk is all zeros.
    pub offset: Vec<u64>,
    /// The absolute file offset of this chunk's stored bytes.
    pub address: u64,
    /// The number of bytes stored at [`address`](Self::address): the filtered
    /// (compressed) size for a filtered dataset, or the raw chunk byte size
    /// otherwise.
    pub storage_size: u64,
    /// Per-filter skip mask: if bit *i* is set, the *i*-th filter of the
    /// pipeline was not applied to this chunk. `0` means every filter applies.
    pub filter_mask: u32,
}

/// A filter's identifier, interpreted parameters, and encoded metadata.
///
/// Obtain filters in application order from
/// [`Dataset::filter_pipeline`](crate::Dataset::filter_pipeline).
/// The reader preserves the recorded name, flags, and client data alongside the interpreted
/// parameters. A staged dataset has [`FilterParameters::Pending`] until its parameters are derived
/// during serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Filter {
    /// The filter identifier encoded in the pipeline message.
    pub id: FilterId,
    /// The recorded filter name, if present.
    pub name: Option<String>,
    /// Whether the writer may omit this filter if it fails during output.
    ///
    /// Each chunk's [`filter_mask`](Chunk::filter_mask) records which filters were omitted.
    pub is_optional: bool,
    /// The interpreted parameters, or their opaque or pending classification.
    pub parameters: FilterParameters,
    flags: u16,
    client_data: Vec<u32>,
}

impl Filter {
    pub(crate) fn pending(id: FilterId, is_optional: bool) -> Self {
        Self {
            id,
            name: None,
            is_optional,
            parameters: FilterParameters::Pending,
            flags: if is_optional {
                crate::filter_pipeline::H5Z_FLAG_OPTIONAL
            } else {
                0
            },
            client_data: Vec::new(),
        }
    }

    pub(crate) fn from_description(
        FilterDescription {
            filter_id,
            name,
            flags,
            client_data,
        }: FilterDescription,
    ) -> Result<Self, FormatError> {
        let id = FilterId::try_from(filter_id)?;
        let parameters = FilterParameters::parse(id, &client_data)?;
        Ok(Self {
            id,
            name,
            is_optional: flags & crate::filter_pipeline::H5Z_FLAG_OPTIONAL != 0,
            parameters,
            flags,
            client_data,
        })
    }

    /// Returns the filter flags, including reserved bits read from the file.
    pub fn flags(&self) -> u16 {
        self.flags
    }

    /// Returns the encoded filter parameters in their stored order.
    ///
    /// Returns an empty slice for [`FilterParameters::Pending`].
    pub fn client_data(&self) -> &[u32] {
        &self.client_data
    }
}

/// The interpreted parameters or interpretation state of a [`Filter`].
///
/// The reader checks the parameter counts and values of the interpreted filters. The complete
/// encoded parameter vector is available through [`Filter::client_data`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FilterParameters {
    /// The Deflate compression level.
    Deflate(DeflateLevel),
    /// The number of bytes per element for byte shuffling.
    Shuffle(NonZeroU32),
    /// A Fletcher32 checksum, which has no client-data values.
    Fletcher32,
    /// The integer or decimal floating-point scale-offset mode.
    ScaleOffset(ScaleOffsetParameters),
    /// Raw parameters for a filter or mode that this API does not interpret.
    Opaque,
    /// Parameters awaiting serialization of a staged dataset.
    Pending,
}

impl FilterParameters {
    fn parse(id: FilterId, values: &[u32]) -> Result<Self, FormatError> {
        Ok(match id {
            FilterId::Deflate => {
                let [level] = values else {
                    return Err(FormatError::InvalidFilterParameterCount {
                        filter_id: id.get(),
                        expected: 1,
                        actual: values.len(),
                    });
                };
                Self::Deflate(DeflateLevel::try_from(*level)?)
            }
            FilterId::Shuffle => {
                let [size] = values else {
                    return Err(FormatError::InvalidFilterParameterCount {
                        filter_id: id.get(),
                        expected: 1,
                        actual: values.len(),
                    });
                };
                Self::Shuffle(
                    NonZeroU32::new(*size)
                        .ok_or(FormatError::InvalidShuffleElementSize { size: *size })?,
                )
            }
            FilterId::Fletcher32 => {
                if !values.is_empty() {
                    return Err(FormatError::InvalidFilterParameterCount {
                        filter_id: id.get(),
                        expected: 0,
                        actual: values.len(),
                    });
                }
                Self::Fletcher32
            }
            FilterId::ScaleOffset => {
                let mode = hdf5_pure_filter::__private::parse_scale_offset_parameters(values)?;
                match mode {
                    Some(mode) => Self::ScaleOffset(ScaleOffsetParameters {
                        mode: ScaleOffset::from_filter_mode(mode),
                    }),
                    None => Self::Opaque,
                }
            }
            _ => Self::Opaque,
        })
    }
}

/// The interpreted mode of a stored scale-offset filter.
///
/// The complete parameter vector is available through [`Filter::client_data`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScaleOffsetParameters {
    mode: ScaleOffset,
}

impl ScaleOffsetParameters {
    /// Returns the stored integer precision or decimal scaling mode.
    pub fn mode(&self) -> ScaleOffset {
        self.mode
    }
}

/// A nonzero identifier stored in an HDF5 filter pipeline.
///
/// Identifiers are available independently of codec features. Versions 1 and 2 of the filter
/// pipeline message are defined in "The Data Storage - Filter Pipeline Message" of the [format
/// specification, version 4.0][spec]. The reader preserves identifiers without a named variant in
/// [`Custom`](Self::Custom).
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FilterId {
    /// Deflate compression.
    Deflate,
    /// Shuffling of data-element bytes.
    Shuffle,
    /// Fletcher32 checksums.
    Fletcher32,
    /// SZIP compression.
    Szip,
    /// N-bit packing.
    Nbit,
    /// Scale-offset compression.
    ScaleOffset,
    /// LZF compression.
    Lzf,
    /// ZFP compression.
    Zfp,
    /// An identifier without a named variant in this enum.
    Custom(CustomFilterId),
}

impl FilterId {
    /// Returns the identifier encoded in the filter pipeline message.
    pub const fn get(self) -> u16 {
        match self {
            Self::Deflate => FILTER_DEFLATE,
            Self::Shuffle => FILTER_SHUFFLE,
            Self::Fletcher32 => FILTER_FLETCHER32,
            Self::Szip => FILTER_SZIP,
            Self::Nbit => FILTER_NBIT,
            Self::ScaleOffset => FILTER_SCALEOFFSET,
            Self::Lzf => FILTER_LZF,
            Self::Zfp => FILTER_ZFP,
            Self::Custom(id) => id.get(),
        }
    }
    /// Returns the registered filter name, or [`None`] for [`Custom`](Self::Custom).
    fn name(self) -> Option<&'static str> {
        Some(match self {
            Self::Deflate => "deflate",
            Self::Shuffle => "shuffle",
            Self::Fletcher32 => "fletcher32",
            Self::Szip => "szip",
            Self::Nbit => "nbit",
            Self::ScaleOffset => "scaleoffset",
            Self::Lzf => "lzf",
            Self::Zfp => "zfp",
            Self::Custom(_) => return None,
        })
    }
}

impl TryFrom<u16> for FilterId {
    type Error = FormatError;

    /// Parses an encoded filter identifier.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFilterId`] if `id` is zero.
    fn try_from(id: u16) -> Result<Self, FormatError> {
        Ok(match id {
            FILTER_RESERVED => return Err(FormatError::InvalidFilterId { id }),
            FILTER_DEFLATE => Self::Deflate,
            FILTER_SHUFFLE => Self::Shuffle,
            FILTER_FLETCHER32 => Self::Fletcher32,
            FILTER_SZIP => Self::Szip,
            FILTER_NBIT => Self::Nbit,
            FILTER_SCALEOFFSET => Self::ScaleOffset,
            FILTER_LZF => Self::Lzf,
            FILTER_ZFP => Self::Zfp,
            id => Self::Custom(CustomFilterId(id)),
        })
    }
}

/// A nonzero filter identifier without a named [`FilterId`] variant.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CustomFilterId(u16);

impl CustomFilterId {
    /// Returns the identifier encoded in the filter pipeline message.
    pub const fn get(self) -> u16 {
        self.0
    }
}

impl TryFrom<u16> for CustomFilterId {
    type Error = FormatError;

    /// Parses an identifier without a named [`FilterId`] variant.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFilterId`] if `id` is zero, and
    /// [`FormatError::InvalidCustomFilterId`] if `id` has a named [`FilterId`] variant.
    fn try_from(id: u16) -> Result<Self, FormatError> {
        match FilterId::try_from(id)? {
            FilterId::Custom(id) => Ok(id),
            _ => Err(FormatError::InvalidCustomFilterId { id }),
        }
    }
}

// ---- Display ----
//
// A caller prints these to describe a dataset, so `Display` is the one-line
// form. `Debug` keeps the full record.

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compact { size } => write!(f, "compact ({size} bytes)"),
            Self::Contiguous {
                address: Some(address),
                size,
            } => write!(f, "contiguous ({size} bytes at 0x{address:x})"),
            Self::Contiguous {
                address: None,
                size,
            } => write!(f, "contiguous ({size} bytes, unallocated)"),
            Self::Chunked { chunk_shape, index } => {
                write!(f, "chunked ({}, {index} index)", Dims(chunk_shape))
            }
            Self::Virtual => f.write_str("virtual"),
        }
    }
}

impl fmt::Display for ChunkIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::BTreeV1 => "B-tree v1",
            Self::SingleChunk => "single chunk",
            Self::Implicit => "implicit",
            Self::FixedArray => "fixed array",
            Self::ExtensibleArray => "extensible array",
            Self::BTreeV2 => "B-tree v2",
        })
    }
}

impl fmt::Display for Filter {
    /// The filter's name and its client data, as `deflate(6)`. A filter this
    /// crate does not name carries its identifier in the same parentheses —
    /// `custom(id=40000)` — so the parentheses hold the filter's parameters and
    /// nothing else.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let well_known = self.id.name();
        match well_known {
            Some(name) => f.write_str(name)?,
            // Recorded by the file, so escaped for the same reason a compound's
            // member name is.
            None => write!(
                f,
                "{}",
                EscapedName(self.name.as_deref().unwrap_or("filter"))
            )?,
        }

        // An unnamed identifier is a parameter of its own, and comes first.
        let write_id = well_known.is_none();
        if write_id || !self.client_data.is_empty() {
            f.write_str("(")?;
            if write_id {
                write!(f, "id={}", self.id.get())?;
            }
            for (i, value) in self.client_data.iter().enumerate() {
                if write_id || i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{value}")?;
            }
            f.write_str(")")?;
        }

        if self.parameters == FilterParameters::Pending {
            f.write_str(" [pending]")?;
        }
        if self.is_optional {
            f.write_str(" [optional]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::address::StoredAddress;
    use crate::address::StoredAddressExt;

    #[rstest]
    #[case(1, FilterId::Deflate)]
    #[case(2, FilterId::Shuffle)]
    #[case(3, FilterId::Fletcher32)]
    #[case(4, FilterId::Szip)]
    #[case(5, FilterId::Nbit)]
    #[case(6, FilterId::ScaleOffset)]
    #[case(32000, FilterId::Lzf)]
    #[case(32013, FilterId::Zfp)]
    fn parses_named_ids(#[case] raw: u16, #[case] id: FilterId) {
        assert_eq!(FilterId::try_from(raw).unwrap(), id);
        assert_eq!(id.get(), raw);
        assert_eq!(
            CustomFilterId::try_from(raw).unwrap_err(),
            FormatError::InvalidCustomFilterId { id: raw }
        );
    }

    #[rstest]
    #[case(7)]
    #[case(255)]
    #[case(256)]
    #[case(32768)]
    #[case(u16::MAX)]
    fn preserves_custom_ids(#[case] raw: u16) {
        let custom = CustomFilterId::try_from(raw).unwrap();
        assert_eq!(custom.get(), raw);
        let id = FilterId::try_from(raw).unwrap();
        assert_eq!(id, FilterId::Custom(custom));
        assert_eq!(id.get(), raw);
    }

    #[test]
    fn rejects_reserved_id() {
        assert_eq!(
            FilterId::try_from(0).unwrap_err(),
            FormatError::InvalidFilterId { id: 0 }
        );
        assert_eq!(
            CustomFilterId::try_from(0).unwrap_err(),
            FormatError::InvalidFilterId { id: 0 }
        );
    }

    #[rstest]
    #[case(FilterId::Deflate, vec![0], FilterParameters::Deflate(DeflateLevel::Zero))]
    #[case(FilterId::Deflate, vec![6], FilterParameters::Deflate(DeflateLevel::Six))]
    #[case(FilterId::Deflate, vec![9], FilterParameters::Deflate(DeflateLevel::Nine))]
    #[case(FilterId::Shuffle, vec![4], FilterParameters::Shuffle(NonZeroU32::new(4).unwrap()))]
    #[case(FilterId::Fletcher32, vec![], FilterParameters::Fletcher32)]
    #[case(FilterId::ScaleOffset, scale_offset_values(), FilterParameters::ScaleOffset(ScaleOffsetParameters { mode: ScaleOffset::Integer(0) }))]
    #[case(FilterId::Lzf, vec![4, 0x0105, 256], FilterParameters::Opaque)]
    #[case(FilterId::Zfp, vec![1, 2, 3, 4, 5, 6], FilterParameters::Opaque)]
    #[case(FilterId::Szip, vec![1, 2, 3, 4], FilterParameters::Opaque)]
    #[case(FilterId::Nbit, vec![1, 2], FilterParameters::Opaque)]
    #[case(FilterId::try_from(40000).unwrap(), vec![u32::MAX, 0, 7], FilterParameters::Opaque)]
    fn interprets_parameters_without_losing_wire_metadata(
        #[case] id: FilterId,
        #[case] values: Vec<u32>,
        #[case] parameters: FilterParameters,
        #[values(0x8000, 0x8001)] flags: u16,
    ) {
        let filter = Filter::from_description(FilterDescription {
            filter_id: id.get(),
            name: Some("recorded filter name".into()),
            flags,
            client_data: values.clone(),
        })
        .unwrap();
        assert_eq!(filter.id, id);
        assert_eq!(filter.parameters, parameters);
        assert_eq!(filter.flags(), flags);
        assert_eq!(
            filter.is_optional,
            flags & crate::filter_pipeline::H5Z_FLAG_OPTIONAL != 0
        );
        assert_eq!(filter.name.as_deref(), Some("recorded filter name"));
        assert_eq!(filter.client_data(), values);
    }

    #[rstest]
    #[case(FilterId::Deflate, vec![], FormatError::InvalidFilterParameterCount { filter_id: FILTER_DEFLATE, expected: 1, actual: 0 })]
    #[case(FilterId::Deflate, vec![6, 7], FormatError::InvalidFilterParameterCount { filter_id: FILTER_DEFLATE, expected: 1, actual: 2 })]
    #[case(FilterId::Shuffle, vec![], FormatError::InvalidFilterParameterCount { filter_id: FILTER_SHUFFLE, expected: 1, actual: 0 })]
    #[case(FilterId::Shuffle, vec![4, 8], FormatError::InvalidFilterParameterCount { filter_id: FILTER_SHUFFLE, expected: 1, actual: 2 })]
    #[case(FilterId::Shuffle, vec![0], FormatError::InvalidShuffleElementSize { size: 0 })]
    #[case(FilterId::Fletcher32, vec![1], FormatError::InvalidFilterParameterCount { filter_id: FILTER_FLETCHER32, expected: 0, actual: 1 })]
    #[case(FilterId::ScaleOffset, vec![2, 0], FormatError::InvalidFilterParameterCount { filter_id: FILTER_SCALEOFFSET, expected: 20, actual: 2 })]
    fn rejects_malformed_known_parameters(
        #[case] id: FilterId,
        #[case] values: Vec<u32>,
        #[case] expected: FormatError,
    ) {
        assert_eq!(FilterParameters::parse(id, &values).unwrap_err(), expected);
    }

    #[rstest]
    #[case(10)]
    #[case(256)]
    #[case(u32::MAX)]
    fn rejects_invalid_deflate_metadata(#[case] level: u32) {
        assert_eq!(
            FilterParameters::parse(FilterId::Deflate, &[level]).unwrap_err(),
            FormatError::InvalidDeflateLevel { level }
        );
    }

    #[rstest]
    #[case(1, 33, FormatError::InvalidScaleOffsetIntegerPrecision { bits: 33, element_size: 4 })]
    #[case(2, 0, FormatError::InvalidScaleOffsetElementCount { count: 0 })]
    #[case(3, 2, FormatError::InvalidScaleOffsetMode { mode: 2, class: 2 })]
    #[case(4, 0, FormatError::InvalidScaleOffsetElementSize { size: 0 })]
    #[case(4, 9, FormatError::InvalidScaleOffsetElementSize { size: 9 })]
    #[case(5, 2, FormatError::InvalidScaleOffsetIntegerSign { sign: 2 })]
    #[case(6, 2, FormatError::InvalidScaleOffsetByteOrder { order: 2 })]
    #[case(7, 2, FormatError::InvalidScaleOffsetFillAvailability { value: 2 })]
    fn rejects_invalid_scale_offset_fields(
        #[case] index: usize,
        #[case] value: u32,
        #[case] expected: FormatError,
    ) {
        let mut values = scale_offset_values();
        values[index] = value;
        assert_eq!(
            FilterParameters::parse(FilterId::ScaleOffset, &values).unwrap_err(),
            expected
        );
    }

    #[rstest]
    #[case(1)]
    #[case(3)]
    #[case(5)]
    fn rejects_invalid_scale_offset_float_size(#[case] size: u32) {
        let mut values = scale_offset_values();
        values[0] = 0;
        values[3] = 1;
        values[4] = size;
        assert_eq!(
            FilterParameters::parse(FilterId::ScaleOffset, &values).unwrap_err(),
            FormatError::InvalidScaleOffsetFloatSize { size }
        );
    }

    #[rstest]
    #[case(0, Some(ScaleOffset::FloatDScale(-2)))]
    #[case(1, None)]
    fn distinguishes_supported_scale_offset_modes(
        #[case] mode: u32,
        #[case] expected: Option<ScaleOffset>,
        #[values(0, u32::MAX)] sign: u32,
    ) {
        let mut values = scale_offset_values();
        values[0] = mode;
        values[1] = u32::from_ne_bytes((-2_i32).to_ne_bytes());
        values[3] = 1;
        values[5] = sign;
        let parsed = FilterParameters::parse(FilterId::ScaleOffset, &values).unwrap();
        match (parsed, expected) {
            (FilterParameters::ScaleOffset(parameters), Some(mode)) => {
                assert_eq!(parameters.mode(), mode)
            }
            (FilterParameters::Opaque, None) => {}
            (parsed, expected) => panic!("expected {expected:?}, got {parsed:?}"),
        }
    }

    fn scale_offset_values() -> Vec<u32> {
        let mut values = vec![0; 20];
        values[..8].copy_from_slice(&[2, 0, 4, 0, 4, 1, 0, 1]);
        values
    }

    #[rstest]
    #[case(ChunkIndexLayout::BTreeV1 { address: None }, ChunkIndex::BTreeV1)]
    #[case(
        ChunkIndexLayout::SingleChunk { filtered: None, address: Some(StoredAddress::new(0x100)) },
        ChunkIndex::SingleChunk
    )]
    #[case(ChunkIndexLayout::Implicit { address: Some(StoredAddress::new(0x100)) }, ChunkIndex::Implicit)]
    #[case(ChunkIndexLayout::FixedArray { address: Some(StoredAddress::new(0x100)) }, ChunkIndex::FixedArray)]
    #[case(
        ChunkIndexLayout::ExtensibleArray { address: Some(StoredAddress::new(0x100)) },
        ChunkIndex::ExtensibleArray
    )]
    #[case(ChunkIndexLayout::BTreeV2 { address: Some(StoredAddress::new(0x100)) }, ChunkIndex::BTreeV2)]
    fn every_parsed_chunk_index_maps_to_a_kind(
        #[case] parsed: ChunkIndexLayout,
        #[case] kind: ChunkIndex,
    ) {
        assert_eq!(ChunkIndex::from_layout(parsed), kind);
    }

    #[test]
    fn only_extensible_array_supports_inplace_append() {
        assert!(ChunkIndex::ExtensibleArray.supports_inplace_append());
        for idx in [
            ChunkIndex::BTreeV1,
            ChunkIndex::SingleChunk,
            ChunkIndex::Implicit,
            ChunkIndex::FixedArray,
            ChunkIndex::BTreeV2,
        ] {
            assert!(!idx.supports_inplace_append());
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod display_tests {
    use super::*;

    #[test]
    fn a_layout_reads_as_one_line() {
        assert_eq!(
            Layout::Compact { size: 40 }.to_string(),
            "compact (40 bytes)"
        );
        assert_eq!(
            Layout::Contiguous {
                address: Some(0x2a0),
                size: 128,
            }
            .to_string(),
            "contiguous (128 bytes at 0x2a0)"
        );
        assert_eq!(
            Layout::Chunked {
                chunk_shape: vec![4, 8],
                index: ChunkIndex::ExtensibleArray,
            }
            .to_string(),
            "chunked (4x8, extensible array index)"
        );
    }

    /// An unallocated dataset says so, rather than printing `None`.
    #[test]
    fn an_unallocated_contiguous_dataset_says_so() {
        let layout = Layout::Contiguous {
            address: None,
            size: 64,
        };
        let shown = layout.to_string();
        assert_eq!(shown, "contiguous (64 bytes, unallocated)");
        assert!(!shown.contains("None"));
    }

    /// Most built-in filters record no name, which would leave a bare number.
    #[test]
    fn a_filter_is_named_by_its_identifier_when_the_file_records_none() {
        let deflate = Filter::from_description(FilterDescription {
            filter_id: 1,
            name: None,
            flags: 0,
            client_data: vec![6],
        })
        .unwrap();
        assert_eq!(deflate.to_string(), "deflate(6)");

        let lzf = Filter::from_description(FilterDescription {
            filter_id: 32000,
            name: None,
            flags: 0,
            client_data: vec![],
        })
        .unwrap();
        assert_eq!(lzf.to_string(), "lzf");
    }

    #[test]
    fn an_unregistered_filter_falls_back_to_its_recorded_name_then_its_id() {
        let named = Filter::from_description(FilterDescription {
            filter_id: 40000,
            name: Some("custom".into()),
            flags: crate::filter_pipeline::H5Z_FLAG_OPTIONAL,
            client_data: vec![],
        })
        .unwrap();
        assert_eq!(named.to_string(), "custom(id=40000) [optional]");

        let anonymous = Filter::from_description(FilterDescription {
            filter_id: 40001,
            name: None,
            flags: 0,
            client_data: vec![],
        })
        .unwrap();
        assert_eq!(anonymous.to_string(), "filter(id=40001)");
    }

    /// The file records this name, so it cannot reach a message unescaped.
    #[test]
    fn a_recorded_filter_name_cannot_carry_a_control_character() {
        let hostile = Filter::from_description(FilterDescription {
            filter_id: 40000,
            name: Some("evil\u{1b}[31m\nname".into()),
            flags: 0,
            client_data: vec![],
        })
        .unwrap();
        let shown = hostile.to_string();
        assert!(!shown.chars().any(char::is_control), "{shown}");
        assert_eq!(shown, "evil\\u{1b}[31m\\nname(id=40000)");
    }

    #[cfg(feature = "zfp")]
    #[test]
    fn the_zfp_name_is_reached_through_its_own_identifier() {
        assert_eq!(FilterId::Zfp.name(), Some("zfp"));
    }

    /// The identifier is labeled, so it cannot read as one of the client-data
    /// values it sits beside.
    #[test]
    fn an_unregistered_filter_keeps_its_id_apart_from_its_client_data() {
        let named = Filter::from_description(FilterDescription {
            filter_id: 40000,
            name: Some("custom".into()),
            flags: 0,
            client_data: vec![7, 8],
        })
        .unwrap();
        assert_eq!(named.to_string(), "custom(id=40000, 7, 8)");
    }
}
