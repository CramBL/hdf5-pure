//! Dataset storage sources and deferred element references.

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::chunked_write::StorageAllocation;
use crate::error::FormatError;
use crate::type_builders::{ProducedPayload, RawChunkPayload};

/// A dataset's single storage source after parsing its builder request.
///
/// An unallocated schema and a payload source are mutually exclusive. An empty dataset
/// without an explicit storage request is represented by an empty in-memory buffer.
pub(crate) enum PreparedStorage {
    /// Encoded element bytes held in memory.
    Bytes(Vec<u8>),
    /// Encoded chunks read verbatim from a provider during emission.
    Chunks(RawChunkPayload),
    /// Contiguous element bytes read from a provider during emission.
    Produced(ProducedPayload),
    /// A schema without element storage.
    Unallocated,
}

impl PreparedStorage {
    /// Parses a single storage source or an unallocated schema.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ConflictingDatasetStorage`] if more than one source is requested
    /// or unallocated storage is paired with a payload. Returns
    /// [`FormatError::DatasetMissingData`] if a nonempty allocated dataset has no source.
    /// Propagates an error staged in `data`.
    pub(crate) fn parse(
        data: Option<Result<Vec<u8>, Box<FormatError>>>,
        chunks: Option<RawChunkPayload>,
        produced: Option<ProducedPayload>,
        allocation: StorageAllocation,
        empty: bool,
    ) -> Result<Self, FormatError> {
        let data = data.transpose().map_err(|error| *error)?;
        match (allocation, data, chunks, produced) {
            (StorageAllocation::Unallocated, None, None, None) => Ok(Self::Unallocated),
            (StorageAllocation::Allocated, Some(bytes), None, None) => Ok(Self::Bytes(bytes)),
            (StorageAllocation::Allocated, None, Some(chunks), None) => Ok(Self::Chunks(chunks)),
            (StorageAllocation::Allocated, None, None, Some(produced)) => {
                Ok(Self::Produced(produced))
            }
            (StorageAllocation::Allocated, None, None, None) if empty => {
                Ok(Self::Bytes(Vec::new()))
            }
            (StorageAllocation::Allocated, None, None, None) => {
                Err(FormatError::DatasetMissingData)
            }
            _ => Err(FormatError::ConflictingDatasetStorage),
        }
    }

    /// Returns the in-memory element bytes for reference patching.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the dataset's storage is not an in-memory buffer.
    pub(crate) fn raw_mut(&mut self) -> Result<&mut Vec<u8>, FormatError> {
        match self {
            Self::Bytes(bytes) => Ok(bytes),
            Self::Chunks(_) | Self::Produced(_) | Self::Unallocated => Err(FormatError::Internal(
                "element patching requires an in-memory dataset".into(),
            )),
        }
    }

    /// Removes the in-memory buffer, or returns an empty buffer for another storage kind.
    pub(crate) fn take_raw(&mut self) -> Vec<u8> {
        match self {
            Self::Bytes(bytes) => core::mem::take(bytes),
            Self::Chunks(_) | Self::Produced(_) | Self::Unallocated => Vec::new(),
        }
    }

    /// Returns the in-memory element bytes, or an empty slice for another storage kind.
    pub(crate) fn raw(&self) -> &[u8] {
        match self {
            Self::Bytes(bytes) => bytes,
            Self::Chunks(_) | Self::Produced(_) | Self::Unallocated => &[],
        }
    }

    pub(crate) fn chunks(&self) -> Option<&RawChunkPayload> {
        match self {
            Self::Chunks(chunks) => Some(chunks),
            Self::Bytes(_) | Self::Produced(_) | Self::Unallocated => None,
        }
    }

    pub(crate) fn produced(&self) -> Option<&ProducedPayload> {
        match self {
            Self::Produced(produced) => Some(produced),
            Self::Bytes(_) | Self::Chunks(_) | Self::Unallocated => None,
        }
    }

    pub(crate) fn allocation(&self) -> StorageAllocation {
        match self {
            Self::Unallocated => StorageAllocation::Unallocated,
            Self::Bytes(_) | Self::Chunks(_) | Self::Produced(_) => StorageAllocation::Allocated,
        }
    }
}
