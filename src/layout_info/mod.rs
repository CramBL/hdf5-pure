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

use alloc::sync::Arc;

use core::fmt;
use core::num::NonZeroU32;

use hdf5_pure_core::__private::PipelineFilterMask;
#[cfg(test)]
use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;

use hdf5_pure_filter::__private::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZF, FILTER_NBIT, FILTER_RESERVED,
    FILTER_SCALEOFFSET, FILTER_SHUFFLE, FILTER_SZIP, FILTER_ZFP,
};

use crate::DeflateLevel;
use crate::FilterMask;
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
/// Obtain chunks from [`Dataset::chunks`](crate::Dataset::chunks), which reads the chunk index
/// and filter metadata without reading chunk data. Each chunk retains the originating dataset's
/// ordered pipeline. [`Self::filters`] reports which filters were applied, skipped, or bypassed
/// for a raw partial edge chunk. [`Self::filter_mask`] returns the recorded bits separately.
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
    /// The recorded skip bits paired with the retained pipeline's length.
    filter_mask: PipelineFilterMask,
    filters: Arc<[Filter]>,
    bypassed_raw: bool,
}

impl Chunk {
    pub(crate) fn new(
        filters: Arc<[Filter]>,
        offset: Vec<u64>,
        address: u64,
        storage_size: u64,
        recorded_mask: u32,
        bypassed_raw: bool,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            filter_mask: PipelineFilterMask::from_raw_bits(recorded_mask, filters.len())?,
            filters,
            offset,
            address,
            storage_size,
            bypassed_raw,
        })
    }

    /// Returns the recorded skip bits for interoperability and inspection.
    ///
    /// Use [`Self::filters`] for filter dispositions. A zero mask may accompany a raw chunk
    /// whose layout bypasses the pipeline.
    pub fn filter_mask(&self) -> FilterMask {
        self.filter_mask.recorded_mask()
    }

    /// Returns filter dispositions bound to this chunk's originating pipeline.
    pub fn filters(&self) -> ChunkFilterView<'_> {
        ChunkFilterView {
            filters: &self.filters,
            mask: self.filter_mask,
            bypassed_raw: self.bypassed_raw,
        }
    }
}

/// The ordered filters and decoding dispositions of one stored chunk.
///
/// Obtain this view through [`Chunk::filters`]. The chunk retains its dataset's pipeline, so
/// the filter order is preserved after the dataset is dropped. Raw partial edge chunks report
/// [`FilterDisposition::BypassedRaw`] for every filter, independently of the recorded mask.
#[derive(Clone, Copy, Debug)]
pub struct ChunkFilterView<'a> {
    filters: &'a [Filter],
    mask: PipelineFilterMask,
    bypassed_raw: bool,
}

impl<'a> ChunkFilterView<'a> {
    /// Iterates over the dataset's filters and their dispositions in application order.
    pub fn dispositions(
        self,
    ) -> impl DoubleEndedIterator<Item = (&'a Filter, FilterDisposition)> + ExactSizeIterator + 'a
    {
        self.filters
            .iter()
            .zip(self.mask.positions())
            .map(move |(filter, (_, skipped))| (filter, self.disposition(skipped)))
    }

    /// Returns the filter and disposition at `index`, or `None` outside the pipeline.
    ///
    /// The index is a position in application order, independent of the filter identifier.
    pub fn filter(&self, index: usize) -> Option<(&'a Filter, FilterDisposition)> {
        self.filters
            .get(index)
            .zip(self.mask.is_skipped(index))
            .map(|(filter, skipped)| (filter, self.disposition(skipped)))
    }

    /// Returns recorded mask bits outside the dataset's pipeline.
    ///
    /// These bits are preserved for inspection and ignored during decoding.
    pub fn uninterpreted_bits(&self) -> u32 {
        self.mask.uninterpreted_bits()
    }

    /// Returns whether the chunk's layout requires raw decoding without any filters.
    pub fn bypasses_pipeline(&self) -> bool {
        self.bypassed_raw
    }

    fn disposition(&self, skipped: bool) -> FilterDisposition {
        if self.bypassed_raw {
            FilterDisposition::BypassedRaw
        } else if skipped {
            FilterDisposition::Skipped
        } else {
            FilterDisposition::Applied
        }
    }
}

/// How a filter participates in decoding one stored chunk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FilterDisposition {
    /// The chunk uses pipeline decoding and this filter's skip bit is clear.
    Applied,
    /// The chunk is a raw partial edge chunk, so decoding bypasses the entire pipeline.
    BypassedRaw,
    /// The chunk uses pipeline decoding and this filter's skip bit is set.
    Skipped,
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
    /// Inspect each chunk's disposition through [`Chunk::filters`].
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
/// The reader checks the parameter counts and values of the interpreted filters. Match the
/// interpretations needed by the application, with a fallback to [`Filter::client_data`] for
/// the complete encoded parameter vector.
#[non_exhaustive]
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
/// Use [`Self::get`] to compare numeric identifiers independently of their classification as
/// named or [`Custom`](Self::Custom) variants.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
#[non_exhaustive]
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
mod tests;
