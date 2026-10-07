//! The per-dataset properties every raw read is performed against.
//!
//! Reading a dataset's element bytes takes five facts about the dataset — where
//! its storage is, what shape it has, what an element is, what filters its bytes
//! passed through, and what unallocated storage reads as — and the call chain
//! that does it is four layers deep: the reader picks a backend, `data_read`
//! dispatches on the layout, `chunked_read` walks the chunk index, and the
//! assembly kernel places what it decoded. Every layer needs all five, so before
//! this type they were threaded through as five positional parameters and the
//! signatures reached twelve arguments.
//!
//! The cost of that shape was not the width; it was that adding a property meant
//! editing roughly thirty call sites, several of them mechanically across test
//! fixtures, where a default inserted in place of a real value compiles and
//! passes (issue #294). A property added to [`RawReadSpec`] instead reaches every
//! layer at once, and the sites that must decide what it is are the few that
//! build the spec.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::convert::Narrow;
use crate::data_layout::{ChunkIndexLayout, ChunkedLayoutFlags, DataLayout};
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::fill_value::FillPattern;
use crate::filter_pipeline::FilterPipeline;

/// What a raw read needs to know about the dataset it is reading.
///
/// These are properties of the *dataset*. The file-level address widths
/// (`offset_size`, `length_size`) travel beside a spec rather than in it: they
/// are the same for every dataset in a file, and the rest of the crate already
/// passes them as a pair.
///
/// `Copy`, so a spec is handed down the call chain the way the five borrows it
/// replaced were, with no clone and no lifetime beyond the borrows themselves.
#[derive(Clone, Copy)]
pub(crate) struct RawReadSpec<'a> {
    /// Storage properties after dataset-level consistency has been established.
    storage: RawReadStorage<'a>,
    /// The dataset's shape, which fixes how many elements the read produces.
    dataspace: &'a Dataspace,
    /// The stored element type, which fixes each element's width.
    datatype: &'a Datatype,
    /// The filter pipeline stored bytes passed through, if any.
    pipeline: Option<&'a FilterPipeline>,
    /// What storage that was never allocated reads as.
    fill: FillPattern<'a>,
    /// Logical dataset byte length derived from the dataspace and datatype.
    byte_len: DatasetByteLen,
}

impl<'a> RawReadSpec<'a> {
    /// Parse dataset properties into the representation used by raw readers.
    ///
    /// For allocated compact and contiguous storage, construction compares the
    /// storage extent with the byte length implied by the dataspace and datatype.
    /// [`RawReadStorage`] has no field for a contiguous extent, so downstream
    /// readers cannot use an unchecked extent after construction succeeds.
    pub(crate) fn parse(
        layout: &'a DataLayout,
        dataspace: &'a Dataspace,
        datatype: &'a Datatype,
        pipeline: Option<&'a FilterPipeline>,
        fill: FillPattern<'a>,
    ) -> Result<Self, FormatError> {
        let byte_len = DatasetByteLen::parse(dataspace, datatype)?;
        let storage = RawReadStorage::parse(layout, byte_len, dataspace.num_elements() != 0)?;
        Ok(Self {
            storage,
            dataspace,
            datatype,
            pipeline,
            fill,
            byte_len,
        })
    }

    /// An unchecked spec for tests that need to drive malformed metadata farther
    /// down the read stack. Live paths can only construct a spec through
    /// [`Self::parse`].
    #[cfg(test)]
    pub(crate) fn plain(
        layout: &'a DataLayout,
        dataspace: &'a Dataspace,
        datatype: &'a Datatype,
    ) -> Self {
        Self {
            storage: RawReadStorage::plain(layout),
            dataspace,
            datatype,
            pipeline: None,
            fill: FillPattern::ZERO,
            byte_len: DatasetByteLen(
                dataspace
                    .num_elements()
                    .saturating_mul(u64::from(datatype.type_size())),
            ),
        }
    }

    pub(crate) fn storage(&self) -> RawReadStorage<'a> {
        self.storage
    }

    pub(crate) fn dataspace(&self) -> &'a Dataspace {
        self.dataspace
    }

    pub(crate) fn datatype(&self) -> &'a Datatype {
        self.datatype
    }

    pub(crate) fn pipeline(&self) -> Option<&'a FilterPipeline> {
        self.pipeline
    }

    pub(crate) fn fill(&self) -> FillPattern<'a> {
        self.fill
    }

    /// The logical dataset byte length, narrowed only when a concrete whole
    /// dataset buffer is about to be materialized.
    pub(crate) fn byte_len(&self) -> Result<usize, FormatError> {
        self.byte_len.to_usize()
    }

    pub(crate) fn byte_len_u64(&self) -> u64 {
        self.byte_len.0
    }

    /// Returns the buffer a dataset whose storage was never allocated reads as.
    ///
    /// A chunked dataset's index is allocated lazily, since the reference C
    /// library leaves the layout message's address undefined until the first
    /// chunk is written, so a dataset that has been created but never written
    /// refers to no index at all. That is neither an error nor an empty read: it
    /// reads as one `fill` element per element of its dataspace, which for a
    /// zero-element dataset is the empty buffer and for any other is a whole
    /// materialized dataset.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnreadableFillValue`] if the dataset's Fill Value
    /// message could not be parsed, leaving those elements no value to read as,
    /// [`FormatError::ZeroSizedDatatype`] if the datatype occupies zero bytes
    /// per element, and [`FormatError::ValueTooLargeForPlatform`] or
    /// [`FormatError::OffsetOverflow`] if the materialized dataset would be
    /// larger than this target can address.
    pub(crate) fn unallocated_buffer(&self) -> Result<Vec<u8>, FormatError> {
        let elem_size = crate::datatype::element_size_usize(self.datatype)?;
        let total = self
            .dataspace
            .num_elements()
            .to_usize()?
            .checked_mul(elem_size.get())
            .ok_or(FormatError::OffsetOverflow {
                offset: self.dataspace.num_elements(),
                length: elem_size.get() as u64,
            })?;
        self.fill.buffer(total)
    }
}

/// Storage facts that remain after [`RawReadSpec::parse`] establishes the
/// dataset-level extent invariant.
///
/// The raw contiguous `size` is deliberately absent. Construction returns an
/// allocated contiguous variant only when that size equals the dataset's logical
/// byte length. Downstream readers therefore have no unchecked storage extent.
#[derive(Clone, Copy)]
pub(crate) enum RawReadStorage<'a> {
    Compact {
        data: &'a [u8],
    },
    Contiguous {
        address: StoredAddress,
    },
    ContiguousUnallocated,
    Chunked {
        flags: ChunkedLayoutFlags,
        chunk_dimensions: &'a [u64],
        index: ChunkIndexLayout,
    },
    Virtual,
}

impl<'a> RawReadStorage<'a> {
    fn parse(
        layout: &'a DataLayout,
        byte_len: DatasetByteLen,
        has_elements: bool,
    ) -> Result<Self, FormatError> {
        Ok(match layout {
            DataLayout::Compact { data } => {
                if has_elements {
                    byte_len.require_storage_extent(data.len() as u64)?;
                }
                Self::Compact { data }
            }
            DataLayout::Contiguous {
                address: Some(address),
                size,
            } => {
                if has_elements {
                    byte_len.require_storage_extent(*size)?;
                }
                Self::Contiguous { address: *address }
            }
            DataLayout::Contiguous { address: None, .. } => Self::ContiguousUnallocated,
            DataLayout::Chunked {
                flags,
                chunk_dimensions,
                index,
            } => Self::Chunked {
                flags: *flags,
                chunk_dimensions,
                index: *index,
            },
            DataLayout::Virtual => Self::Virtual,
        })
    }

    #[cfg(test)]
    fn plain(layout: &'a DataLayout) -> Self {
        match layout {
            DataLayout::Compact { data } => Self::Compact { data },
            DataLayout::Contiguous {
                address: Some(address),
                ..
            } => Self::Contiguous { address: *address },
            DataLayout::Contiguous { address: None, .. } => Self::ContiguousUnallocated,
            DataLayout::Chunked {
                flags,
                chunk_dimensions,
                index,
            } => Self::Chunked {
                flags: *flags,
                chunk_dimensions,
                index: *index,
            },
            DataLayout::Virtual => Self::Virtual,
        }
    }
}

/// Logical bytes implied by a dataset's dataspace and datatype.
#[derive(Clone, Copy)]
struct DatasetByteLen(u64);

impl DatasetByteLen {
    fn parse(dataspace: &Dataspace, datatype: &Datatype) -> Result<Self, FormatError> {
        let num_elements = dataspace.num_elements();
        let elem_size = u64::from(datatype.type_size());
        Ok(Self(num_elements.checked_mul(elem_size).ok_or(
            FormatError::OffsetOverflow {
                offset: num_elements,
                length: elem_size,
            },
        )?))
    }

    fn require_storage_extent(self, actual: u64) -> Result<(), FormatError> {
        if actual == self.0 {
            return Ok(());
        }
        Err(FormatError::DataSizeMismatch {
            expected: self.to_usize()?,
            actual: actual.to_usize()?,
        })
    }

    fn to_usize(self) -> Result<usize, FormatError> {
        self.0.to_usize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataspace::DataspaceType;
    use crate::type_builders::make_i32_type;

    fn four_i32s() -> Dataspace {
        Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![4],
            max_dimensions: None,
        }
    }

    #[test]
    fn compact_extent_mismatch_is_rejected_while_parsing_the_read_spec() {
        let layout = DataLayout::Compact { data: vec![0; 8] };
        let Err(err) = RawReadSpec::parse(
            &layout,
            &four_i32s(),
            &make_i32_type(),
            None,
            FillPattern::ZERO,
        ) else {
            panic!("mismatched compact storage extent was accepted");
        };
        let FormatError::DataSizeMismatch { expected, actual } = err else {
            panic!("expected DataSizeMismatch, got {err:?}");
        };
        assert_eq!(expected, 16);
        assert_eq!(actual, 8);
    }

    #[test]
    fn contiguous_extent_mismatch_is_rejected_while_parsing_the_read_spec() {
        let layout = DataLayout::Contiguous {
            address: Some(StoredAddress::new(128)),
            size: 32,
        };
        let Err(err) = RawReadSpec::parse(
            &layout,
            &four_i32s(),
            &make_i32_type(),
            None,
            FillPattern::ZERO,
        ) else {
            panic!("mismatched contiguous storage extent was accepted");
        };
        let FormatError::DataSizeMismatch { expected, actual } = err else {
            panic!("expected DataSizeMismatch, got {err:?}");
        };
        assert_eq!(expected, 16);
        assert_eq!(actual, 32);
    }
}
