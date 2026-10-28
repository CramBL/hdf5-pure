//! Dataset storage sources and deferred element references.

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::chunked_write::StorageAllocation;
use crate::error::FormatError;
use crate::type_builders::{ObjectRefPatch, ProducedPayload, RawChunkPayload, VlStringStaging};

/// A dataset's single storage source after parsing its builder request.
///
/// Deferred references are stored with their element bytes. An empty allocated dataset with
/// no explicit source has an empty element buffer.
pub(crate) enum PreparedStorage {
    /// Encoded chunks read verbatim from a provider during emission.
    Chunks(RawChunkPayload),
    /// Element bytes or an unallocated declaration.
    Elements(PreparedElements),
    /// Contiguous element bytes read from a provider during emission.
    Produced(ProducedPayload),
}

impl PreparedStorage {
    /// Parses one storage source and its deferred references.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ConflictingDatasetStorage`] if multiple sources are supplied,
    /// unallocated storage is paired with a source, or deferred references lack an element
    /// buffer. Returns [`FormatError::DatasetMissingData`] if a nonempty allocated dataset has
    /// no source. Propagates an error staged in `data`.
    pub(crate) fn parse(
        data: Option<Result<Vec<u8>, Box<FormatError>>>,
        chunks: Option<RawChunkPayload>,
        produced: Option<ProducedPayload>,
        allocation: StorageAllocation,
        empty: bool,
        reference_targets: Option<Vec<ObjectRefPatch>>,
        vl_string_staging: Option<VlStringStaging>,
    ) -> Result<Self, FormatError> {
        let data = data.transpose().map_err(|error| *error)?;
        let deferred = reference_targets.is_some() || vl_string_staging.is_some();
        match (allocation, data, chunks, produced) {
            (StorageAllocation::Unallocated, None, None, None) if !deferred => {
                Ok(Self::Elements(PreparedElements::Unallocated))
            }
            (StorageAllocation::Allocated, Some(raw), None, None) => {
                Ok(Self::Elements(PreparedElements::Bytes(PreparedBytes {
                    raw,
                    reference_targets,
                    vl_string_staging,
                })))
            }
            (StorageAllocation::Allocated, None, Some(chunks), None) if !deferred => {
                Ok(Self::Chunks(chunks))
            }
            (StorageAllocation::Allocated, None, None, Some(produced)) if !deferred => {
                Ok(Self::Produced(produced))
            }
            (StorageAllocation::Allocated, None, None, None) if empty && !deferred => Ok(
                Self::Elements(PreparedElements::Bytes(PreparedBytes::new(Vec::new()))),
            ),
            (StorageAllocation::Allocated, None, None, None) if !deferred => {
                Err(FormatError::DatasetMissingData)
            }
            _ => Err(FormatError::ConflictingDatasetStorage),
        }
    }

    /// Returns the element buffer for reference patching or appending.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if this dataset has no element buffer.
    pub(crate) fn raw_mut(&mut self) -> Result<&mut Vec<u8>, FormatError> {
        self.elements_mut()?.raw_mut()
    }

    /// Takes the unresolved object-reference targets, leaving `None`.
    pub(crate) fn take_reference_targets(&mut self) -> Option<Vec<ObjectRefPatch>> {
        match self {
            Self::Elements(elements) => elements.take_reference_targets(),
            Self::Chunks(_) | Self::Produced(_) => None,
        }
    }

    /// Takes the variable-length heap collections and patch offsets, leaving `None`.
    pub(crate) fn take_vl_string_staging(&mut self) -> Option<VlStringStaging> {
        match self {
            Self::Elements(elements) => elements.take_vl_string_staging(),
            Self::Chunks(_) | Self::Produced(_) => None,
        }
    }

    /// Stores `staging` with this dataset's element bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if this dataset has no element buffer.
    pub(crate) fn restore_vl_string_staging(
        &mut self,
        staging: VlStringStaging,
    ) -> Result<(), FormatError> {
        self.elements_mut()?.bytes_mut()?.vl_string_staging = Some(staging);
        Ok(())
    }

    /// Returns the element bytes, or an empty slice for storage with no element buffer.
    pub(crate) fn raw(&self) -> &[u8] {
        self.elements().map_or(&[], PreparedElements::raw)
    }

    /// Returns the provider and metadata for encoded chunks, if supplied.
    pub(crate) fn chunks(&self) -> Option<&RawChunkPayload> {
        match self {
            Self::Chunks(chunks) => Some(chunks),
            Self::Elements(_) | Self::Produced(_) => None,
        }
    }

    /// Returns the provider and declared lengths for contiguous data, if supplied.
    pub(crate) fn produced(&self) -> Option<&ProducedPayload> {
        match self {
            Self::Produced(produced) => Some(produced),
            Self::Chunks(_) | Self::Elements(_) => None,
        }
    }

    /// Returns the allocation policy for element storage.
    pub(crate) fn allocation(&self) -> StorageAllocation {
        self.elements()
            .map_or(StorageAllocation::Allocated, PreparedElements::allocation)
    }

    /// Returns the unresolved object-reference targets, if present.
    pub(crate) fn reference_targets(&self) -> Option<&[ObjectRefPatch]> {
        self.elements()
            .and_then(PreparedElements::reference_targets)
    }

    /// Returns the variable-length heap collections and patch offsets, if present.
    pub(crate) fn vl_string_staging(&self) -> Option<&VlStringStaging> {
        self.elements()
            .and_then(PreparedElements::vl_string_staging)
    }

    fn elements_mut(&mut self) -> Result<&mut PreparedElements, FormatError> {
        match self {
            Self::Elements(elements) => Ok(elements),
            Self::Chunks(_) | Self::Produced(_) => Err(FormatError::Internal(
                "element patching requires an in-memory dataset".into(),
            )),
        }
    }

    fn elements(&self) -> Option<&PreparedElements> {
        match self {
            Self::Elements(elements) => Some(elements),
            Self::Chunks(_) | Self::Produced(_) => None,
        }
    }
}

/// Element storage supported by in-place dataset creation.
///
/// Providers are represented separately in [`PreparedStorage`].
pub(crate) enum PreparedElements {
    /// An element buffer and its deferred references.
    Bytes(PreparedBytes),
    /// A declaration whose element storage remains unallocated.
    Unallocated,
}

impl PreparedElements {
    /// Returns the element buffer for reference patching or appending.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if this dataset has no element buffer.
    pub(crate) fn raw_mut(&mut self) -> Result<&mut Vec<u8>, FormatError> {
        Ok(&mut self.bytes_mut()?.raw)
    }

    /// Takes the unresolved object-reference targets, leaving `None`.
    pub(crate) fn take_reference_targets(&mut self) -> Option<Vec<ObjectRefPatch>> {
        match self {
            Self::Bytes(bytes) => bytes.reference_targets.take(),
            Self::Unallocated => None,
        }
    }

    /// Takes the variable-length heap collections and patch offsets, leaving `None`.
    pub(crate) fn take_vl_string_staging(&mut self) -> Option<VlStringStaging> {
        match self {
            Self::Bytes(bytes) => bytes.vl_string_staging.take(),
            Self::Unallocated => None,
        }
    }

    /// Returns the element bytes, or an empty slice for storage with no element buffer.
    pub(crate) fn raw(&self) -> &[u8] {
        match self {
            Self::Bytes(bytes) => &bytes.raw,
            Self::Unallocated => &[],
        }
    }

    /// Returns the allocation policy for element storage.
    pub(crate) fn allocation(&self) -> StorageAllocation {
        match self {
            Self::Bytes(_) => StorageAllocation::Allocated,
            Self::Unallocated => StorageAllocation::Unallocated,
        }
    }

    /// Returns the unresolved object-reference targets, if present.
    pub(crate) fn reference_targets(&self) -> Option<&[ObjectRefPatch]> {
        match self {
            Self::Bytes(bytes) => bytes.reference_targets.as_deref(),
            Self::Unallocated => None,
        }
    }

    /// Returns the variable-length heap collections and patch offsets, if present.
    pub(crate) fn vl_string_staging(&self) -> Option<&VlStringStaging> {
        match self {
            Self::Bytes(bytes) => bytes.vl_string_staging.as_ref(),
            Self::Unallocated => None,
        }
    }

    fn bytes_mut(&mut self) -> Result<&mut PreparedBytes, FormatError> {
        match self {
            Self::Bytes(bytes) => Ok(bytes),
            Self::Unallocated => Err(FormatError::Internal(
                "element patching requires an allocated dataset".into(),
            )),
        }
    }
}

/// An element buffer with optional object targets and variable-length heap collections.
pub(crate) struct PreparedBytes {
    raw: Vec<u8>,
    /// Object-reference patches awaiting target-address resolution.
    reference_targets: Option<Vec<ObjectRefPatch>>,
    /// Variable-length reference offsets and heap collections retained for emission.
    vl_string_staging: Option<VlStringStaging>,
}

impl PreparedBytes {
    /// Creates an element buffer whose deferred-reference lists are empty.
    pub(crate) fn new(raw: Vec<u8>) -> Self {
        Self {
            raw,
            reference_targets: None,
            vl_string_staging: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::num::NonZeroUsize;
    use core::panic::AssertUnwindSafe;

    use rstest::rstest;

    use crate::chunked_write::{ChunkProvider, StorageAllocation};
    use crate::data_layout::ChunkedLayoutFlags;
    use crate::error::FormatError;
    use crate::type_builders::{ProducedPayload, RawChunkPayload, VlStringStaging};

    use super::PreparedStorage;

    #[rstest]
    #[case::bytes_and_chunks([true, true, false], StorageAllocation::Allocated, false)]
    #[case::bytes_and_producer([true, false, true], StorageAllocation::Allocated, false)]
    #[case::chunks_and_producer([false, true, true], StorageAllocation::Allocated, false)]
    #[case::all_sources([true, true, true], StorageAllocation::Allocated, false)]
    #[case::unallocated_bytes([true, false, false], StorageAllocation::Unallocated, false)]
    #[case::unallocated_chunks([false, true, false], StorageAllocation::Unallocated, false)]
    #[case::unallocated_producer([false, false, true], StorageAllocation::Unallocated, false)]
    #[case::deferred_chunks([false, true, false], StorageAllocation::Allocated, true)]
    #[case::deferred_producer([false, false, true], StorageAllocation::Allocated, true)]
    #[case::deferred_unallocated([false, false, false], StorageAllocation::Unallocated, true)]
    #[case::deferred_without_bytes([false, false, false], StorageAllocation::Allocated, true)]
    fn conflicting_storage_sources_are_rejected(
        #[case] sources: [bool; 3],
        #[case] allocation: StorageAllocation,
        #[case] deferred: bool,
        #[values(false, true)] empty: bool,
    ) {
        let [bytes, chunks, produced] = sources;
        let result = PreparedStorage::parse(
            bytes.then(|| Ok(Vec::new())),
            chunks.then(raw_chunks),
            produced.then(produced_bytes),
            allocation,
            empty,
            deferred.then(Vec::new),
            None,
        );
        let Err(error) = result else {
            panic!("expected conflicting storage sources");
        };
        assert_eq!(error, FormatError::ConflictingDatasetStorage);
    }

    #[rstest]
    #[case::object_references(true)]
    #[case::variable_length_references(false)]
    fn deferred_references_remain_with_their_element_bytes(#[case] object: bool) {
        let mut storage = PreparedStorage::parse(
            Some(Ok(vec![0; 16])),
            None,
            None,
            StorageAllocation::Allocated,
            false,
            object.then(Vec::new),
            (!object).then(|| VlStringStaging {
                collections: vec![vec![1, 2, 3]],
                patch_offsets: vec![0],
            }),
        )
        .unwrap();
        assert_eq!(storage.raw(), &[0; 16]);
        assert_eq!(storage.allocation(), StorageAllocation::Allocated);
        assert_eq!(storage.reference_targets().is_some(), object);
        assert_eq!(storage.vl_string_staging().is_some(), !object);
        if let Some(staging) = storage.take_vl_string_staging() {
            assert_eq!(staging.collections, [vec![1, 2, 3]]);
            assert_eq!(staging.patch_offsets, [0]);
            storage.raw_mut().unwrap()[0] = 7;
            storage.restore_vl_string_staging(staging).unwrap();
            assert_eq!(storage.raw()[0], 7);
            assert_eq!(storage.vl_string_staging().unwrap().patch_offsets, [0]);
        } else {
            assert_eq!(storage.take_reference_targets().unwrap(), []);
            assert!(storage.reference_targets().is_none());
        }
    }

    #[rstest]
    #[case::allocated(StorageAllocation::Allocated)]
    #[case::unallocated(StorageAllocation::Unallocated)]
    fn empty_storage_preserves_allocation(#[case] allocation: StorageAllocation) {
        let storage =
            PreparedStorage::parse(None, None, None, allocation, true, None, None).unwrap();
        assert_eq!(storage.allocation(), allocation);
        assert_eq!(storage.raw(), b"");
        assert!(storage.reference_targets().is_none());
        assert!(storage.vl_string_staging().is_none());
    }

    struct UnreadProvider;

    impl ChunkProvider for UnreadProvider {
        fn chunk_bytes(&self, _: usize, _: &mut Vec<u8>) -> Result<(), FormatError> {
            Err(FormatError::Internal(
                "request parsing read the provider".into(),
            ))
        }
    }

    fn raw_chunks() -> RawChunkPayload {
        RawChunkPayload {
            chunk_dims: vec![1],
            element_size: NonZeroUsize::new(1).unwrap(),
            flags: ChunkedLayoutFlags::new(0),
            pipeline_message: None,
            meta: Vec::new(),
            provider: AssertUnwindSafe(Box::new(UnreadProvider)),
        }
    }

    fn produced_bytes() -> ProducedPayload {
        ProducedPayload {
            total_bytes: 0,
            block_bytes: 1,
            provider: AssertUnwindSafe(Box::new(UnreadProvider)),
        }
    }
}
