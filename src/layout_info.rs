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

use crate::data_layout::ChunkIndexLayout;
use crate::display::{Dims, EscapedName};
use crate::filter_pipeline::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZF, FILTER_SCALEOFFSET, FILTER_SHUFFLE,
};

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
    /// A version-2 B-tree indexes the chunks (several unlimited dimensions). A
    /// dataset with this index is classified here, but enumerating its chunks
    /// with [`Dataset::chunks`](crate::Dataset::chunks) is not yet supported.
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

/// One filter in a dataset's pipeline.
///
/// The curated per-filter analogue of `H5Pget_filter2`. Obtain the ordered
/// pipeline with [`Dataset::filter_pipeline`](crate::Dataset::filter_pipeline);
/// [`Dataset::filters`](crate::Dataset::filters) stays the lighter call when
/// only the identifiers are needed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Filter {
    /// The registered HDF5 filter identifier, the same numbering returned by
    /// [`Dataset::filters`](crate::Dataset::filters). `Display` names the ones
    /// this crate knows, such as 1 = deflate or 32000 = lzf.
    pub id: u16,
    /// The filter's recorded name, when the file stores one. Absent for most
    /// built-in filters, which are identified by [`id`](Self::id) alone.
    pub name: Option<String>,
    /// Whether the filter is optional. When `true`, a reader that cannot apply
    /// the filter may skip it; a mandatory filter (`false`) must be applied for
    /// the data to decode correctly.
    pub is_optional: bool,
    /// The filter's client data (`cd_values`): the auxiliary parameters stored
    /// with it — for deflate, one value, the compression level. The meaning is
    /// filter-specific.
    pub client_data: Vec<u32>,
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
        let well_known = well_known_filter_name(self.id);
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
                write!(f, "id={}", self.id)?;
            }
            for (i, value) in self.client_data.iter().enumerate() {
                if write_id || i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{value}")?;
            }
            f.write_str(")")?;
        }

        if self.is_optional {
            f.write_str(" [optional]")?;
        }
        Ok(())
    }
}

/// The name of a filter this crate knows by identifier.
///
/// Most built-in filters record no name of their own, which would otherwise
/// leave a bare number in the output. Naming one is not a claim that this crate
/// can run it: a message reporting a filter it cannot decode is exactly where
/// the name earns its keep.
fn well_known_filter_name(id: u16) -> Option<&'static str> {
    Some(match id {
        // The filters this crate implements are matched through their
        // constants, so the two lists cannot drift apart.
        FILTER_DEFLATE => "deflate",
        FILTER_SHUFFLE => "shuffle",
        FILTER_FLETCHER32 => "fletcher32",
        FILTER_SCALEOFFSET => "scaleoffset",
        FILTER_LZF => "lzf",
        // Registered identifiers with no constant here: szip and nbit have no
        // implementation, and `FILTER_ZFP` is behind the `zfp` feature while
        // the name is worth reporting whether or not the decoder is built.
        4 => "szip",
        5 => "nbit",
        32013 => "zfp",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::address::StoredAddress;

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
        let deflate = Filter {
            id: 1,
            name: None,
            is_optional: false,
            client_data: vec![6],
        };
        assert_eq!(deflate.to_string(), "deflate(6)");

        let lzf = Filter {
            id: 32000,
            name: None,
            is_optional: false,
            client_data: vec![],
        };
        assert_eq!(lzf.to_string(), "lzf");
    }

    #[test]
    fn an_unregistered_filter_falls_back_to_its_recorded_name_then_its_id() {
        let named = Filter {
            id: 40000,
            name: Some("custom".into()),
            is_optional: true,
            client_data: vec![],
        };
        assert_eq!(named.to_string(), "custom(id=40000) [optional]");

        let anonymous = Filter {
            id: 40001,
            name: None,
            is_optional: false,
            client_data: vec![],
        };
        assert_eq!(anonymous.to_string(), "filter(id=40001)");
    }

    /// The file records this name, so it cannot reach a message unescaped.
    #[test]
    fn a_recorded_filter_name_cannot_carry_a_control_character() {
        let hostile = Filter {
            id: 40000,
            name: Some("evil\u{1b}[31m\nname".into()),
            is_optional: false,
            client_data: vec![],
        };
        let shown = hostile.to_string();
        assert!(!shown.chars().any(char::is_control), "{shown}");
        assert_eq!(shown, "evil\\u{1b}[31m\\nname(id=40000)");
    }

    /// The `zfp` identifier is written as a literal, its constant being behind
    /// a feature, so it is pinned to that constant here.
    #[cfg(feature = "zfp")]
    #[test]
    fn the_zfp_name_is_reached_through_its_own_identifier() {
        assert_eq!(
            well_known_filter_name(crate::filter_pipeline::FILTER_ZFP),
            Some("zfp")
        );
    }

    /// The identifier is labeled, so it cannot read as one of the client-data
    /// values it sits beside.
    #[test]
    fn an_unregistered_filter_keeps_its_id_apart_from_its_client_data() {
        let named = Filter {
            id: 40000,
            name: Some("custom".into()),
            is_optional: false,
            client_data: vec![7, 8],
        };
        assert_eq!(named.to_string(), "custom(id=40000, 7, 8)");
    }
}
