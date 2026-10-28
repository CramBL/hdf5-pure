//! Dataset requests, backend preparation, and deferred element references.

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::FillTime;
use crate::attribute_request::PreparedAttributes;
use crate::chunked_write::{ChunkOptions, StorageAllocation};
use crate::convert::Narrow;
use crate::dataspace::{Dataspace, DataspaceType, Extent};
use crate::datatype::Datatype;
#[cfg(feature = "std")]
use crate::error::Error;
use crate::error::FormatError;
use crate::file_writer::{MAX_DATASET_RANK, UNLIMITED_DIMENSION};
use crate::object_path::ObjectPathBuf;
#[cfg(feature = "provenance")]
use crate::provenance::Provenance;
use crate::type_builders::{
    AttributeSpecs, DatasetBuilder, ObjectRefPatch, ProducedPayload, RawChunkPayload,
    VlStringStaging,
};

/// A dataset request with a nonzero element width and checked shape, payload, fill, and attributes.
///
/// Explicit creation options are preserved until conversion to a backend operation. Deferred
/// object and heap references remain unresolved.
pub(crate) struct PreparedDataset {
    metadata: DatasetMetadata,
    storage: PreparedStorage,
    attribute_specs: AttributeSpecs,
}

impl PreparedDataset {
    /// Parses the common dataset invariants without resolving file addresses.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DatasetMissingDatatype`] or [`FormatError::DatasetMissingShape`]
    /// if required metadata is absent, [`FormatError::DatasetRankUnsupported`] if the rank exceeds
    /// [`MAX_DATASET_RANK`], or [`FormatError::DatasetDimensionUnlimited`] if a current dimension
    /// is unlimited.
    ///
    /// Returns [`FormatError::DatasetSizeOverflow`] if the byte extent overflows,
    /// [`FormatError::LogicalShapeDataMismatch`] if an allocated element buffer's length or a
    /// producer's declared byte length disagrees with that extent,
    /// [`FormatError::InvalidChunkGeometry`] if chunk dimensions or maxima are incompatible, or
    /// [`FormatError::FillValueSizeMismatch`] if a fill value is not one element wide.
    ///
    /// With the `provenance` feature, returns [`FormatError::DatasetProvenanceRequiresData`] for a
    /// nonempty unallocated declaration with requested checksums. Propagates datatype, storage,
    /// and attribute preparation errors.
    pub(crate) fn parse(builder: DatasetBuilder) -> Result<Self, FormatError> {
        let DatasetBuilder {
            name: _,
            datatype,
            shape,
            maxshape,
            data,
            attrs,
            chunk_options,
            raw_chunks,
            produced,
            reference_targets,
            vl_string_staging,
            fill,
            fill_time,
            allocation,
            committed_datatype_path,
            #[cfg(feature = "provenance")]
            provenance,
        } = builder;
        let dt = datatype.ok_or(FormatError::DatasetMissingDatatype)?;
        let shape = shape.ok_or(FormatError::DatasetMissingShape)?;
        let storage = PreparedStorage::parse(
            data,
            raw_chunks,
            produced,
            allocation,
            shape.contains(&0),
            reference_targets,
            vl_string_staging,
        )?;
        let width = dt.element_size()?;
        let element_size = width.narrow::<NonZeroUsize>()?;
        let rank = match u8::try_from(shape.len()) {
            Ok(rank) if rank <= MAX_DATASET_RANK => rank,
            _ => {
                return Err(FormatError::DatasetRankUnsupported {
                    rank: shape.len(),
                    maximum: MAX_DATASET_RANK,
                });
            }
        };
        if let Some(dimension) = shape.iter().position(|&size| size == UNLIMITED_DIMENSION) {
            return Err(FormatError::DatasetDimensionUnlimited { dimension });
        }
        let extent = if shape.contains(&0) {
            0
        } else {
            shape
                .iter()
                .try_fold(u64::from(width.get()), |size, &dimension| {
                    size.checked_mul(dimension)
                        .ok_or(FormatError::DatasetSizeOverflow { size, dimension })
                })?
        };
        let unallocated = storage.allocation() == StorageAllocation::Unallocated;
        let region_len = storage
            .produced()
            .map_or(storage.raw().len() as u64, |p| p.total_bytes);
        if !unallocated && storage.chunks().is_none() && region_len != extent {
            return Err(FormatError::LogicalShapeDataMismatch {
                expected: extent,
                actual: region_len,
                element_size: width.into(),
            });
        }
        if chunk_options.is_chunked() || maxshape.is_some() {
            Extent::new(&shape, maxshape.as_deref())
                .and_then(|geometry| chunk_options.validate_geometry(geometry))
                .map_err(FormatError::InvalidChunkGeometry)?;
        }
        if let Some(fill) = &fill {
            if fill.len() != element_size.get() {
                return Err(FormatError::FillValueSizeMismatch {
                    expected: element_size.get(),
                    actual: fill.len(),
                });
            }
        }
        #[cfg(feature = "provenance")]
        if unallocated && extent != 0 && provenance.is_some() {
            return Err(FormatError::DatasetProvenanceRequiresData);
        }
        let prepared_attrs = PreparedAttributes::parse(&attrs)?;
        Ok(Self {
            metadata: DatasetMetadata {
                dt,
                element_size,
                ds: Dataspace {
                    space_type: if shape.is_empty() {
                        DataspaceType::Scalar
                    } else {
                        DataspaceType::Simple
                    },
                    rank,
                    dimensions: shape,
                    max_dimensions: maxshape,
                },
                extent,
                attrs: prepared_attrs,
                chunk_options,
                fill,
                fill_time,
                committed_datatype_path,
                #[cfg(feature = "provenance")]
                provenance: provenance.map(|p| Provenance {
                    creator: p.creator,
                    timestamp: p.timestamp,
                    source: p.source,
                }),
            },
            storage,
            attribute_specs: attrs,
        })
    }

    /// Prepares creation with providers and committed datatype paths available to the writer.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::PartialEdgeSkipRequiresChunking`] if the request disables partial
    /// edge filtering on contiguous storage.
    pub(crate) fn into_whole_file(self) -> Result<WholeFileDataset, FormatError> {
        self.metadata
            .chunk_options
            .validate_partial_edge_filtering(self.metadata.ds.max_dimensions.as_deref())?;
        Ok(WholeFileDataset {
            committed_datatype_paths: self.attribute_specs.committed_datatype_paths(),
            metadata: self.metadata,
            storage: self.storage,
        })
    }

    /// Prepares creation using the storage and datatypes supported by the editor.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for providers, committed datatypes, or chunked or
    /// extensible datasets with deferred element references. Returns [`Error::Format`] with
    /// [`FormatError::PartialEdgeSkipRequiresChunking`] if the request disables partial edge
    /// filtering on contiguous storage.
    #[cfg(feature = "std")]
    pub(crate) fn into_in_place(self) -> Result<InPlaceDataset, Error> {
        self.metadata
            .chunk_options
            .validate_partial_edge_filtering(self.metadata.ds.max_dimensions.as_deref())?;
        self.into_editor()
    }

    /// Prepares replacement values after rejecting explicit creation options.
    ///
    /// Compatibility with the existing datatype and shape is checked during commit planning.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for unallocated storage, providers, committed datatypes,
    /// deferred object references, or creation options. An explicitly supplied fill policy is
    /// rejected even when it equals the default.
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if allocated storage has no element
    /// buffer after the capability checks.
    #[cfg(feature = "std")]
    pub(crate) fn into_overwrite(self) -> Result<ValueOverwrite, Error> {
        let Self {
            metadata,
            storage,
            attribute_specs,
        } = self;
        if storage.allocation() == StorageAllocation::Unallocated {
            return Err(Error::EditUnsupported(
                "write_dataset requires element data, not unallocated storage",
            ));
        }
        if metadata.chunk_options.is_chunked() || metadata.ds.max_dimensions.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot make a dataset chunked, filtered, or extensible",
            ));
        }
        if metadata.chunk_options.partial_edge_filtering.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot set whether partial edge chunks are filtered (set it when the dataset is created)",
            ));
        }
        #[cfg(feature = "provenance")]
        let generated_attrs = metadata.provenance.is_some();
        #[cfg(not(feature = "provenance"))]
        let generated_attrs = false;
        if !metadata.attrs.messages.is_empty() || generated_attrs {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot set attributes (set them with a separate edit)",
            ));
        }
        if metadata.fill.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot change the fill value (set it when the dataset is created)",
            ));
        }
        if metadata.fill_time.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot change the fill time (set it when the dataset is created)",
            ));
        }
        if storage.reference_targets().is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset cannot overwrite an object-reference dataset's data in place yet",
            ));
        }
        let editor = Self {
            metadata,
            storage,
            attribute_specs,
        }
        .into_editor()?;
        let PreparedElements::Bytes(bytes) = editor.storage else {
            return Err(FormatError::Internal(
                "an allocated overwrite has no element buffer".into(),
            )
            .into());
        };
        Ok(ValueOverwrite {
            dt: editor.metadata.dt,
            ds: editor.metadata.ds,
            bytes,
        })
    }

    #[cfg(feature = "std")]
    fn into_editor(self) -> Result<InPlaceDataset, Error> {
        let Self {
            metadata,
            storage,
            attribute_specs: _,
        } = self;
        let storage = match storage {
            PreparedStorage::Elements(elements) => elements,
            PreparedStorage::Chunks(_) => {
                return Err(Error::EditUnsupported(
                    "encoded chunk providers cannot be added in place yet",
                ));
            }
            PreparedStorage::Produced(_) => {
                return Err(Error::EditUnsupported(
                    "data producers cannot be added in place yet",
                ));
            }
        };
        let chunked = metadata.chunk_options.is_chunked() || metadata.ds.max_dimensions.is_some();
        if storage.vl_string_staging().is_some() && chunked {
            return Err(Error::EditUnsupported(
                "chunked or extensible variable-length-string datasets cannot be added in place yet",
            ));
        }
        if storage.reference_targets().is_some() && chunked {
            return Err(Error::EditUnsupported(
                "chunked or extensible object-reference datasets cannot be added in place yet",
            ));
        }
        if metadata.committed_datatype_path.is_some()
            || metadata
                .attrs
                .messages
                .iter()
                .any(|a| a.datatype_location.is_committed())
        {
            return Err(Error::EditUnsupported(
                "a dataset or attribute naming a committed (shared) datatype cannot be added in place; write the file with FileBuilder instead",
            ));
        }
        Ok(InPlaceDataset { metadata, storage })
    }
}

/// Common metadata with checked dimensions and explicit fill settings.
pub(crate) struct DatasetMetadata {
    /// The element datatype, including the definition of a requested committed datatype.
    pub(crate) dt: Datatype,
    pub(crate) element_size: NonZeroUsize,
    /// The checked current dimensions and optional compatible maximum dimensions.
    pub(crate) ds: Dataspace,
    /// The logical byte length, including elements whose storage remains unallocated.
    pub(crate) extent: u64,
    pub(crate) attrs: PreparedAttributes,
    pub(crate) chunk_options: ChunkOptions,
    pub(crate) fill: Option<Vec<u8>>,
    /// The explicitly supplied fill policy, before default normalization.
    pub(crate) fill_time: Option<FillTime>,
    /// The committed datatype path awaiting address resolution, if supplied.
    pub(crate) committed_datatype_path: Option<ObjectPathBuf>,
    /// Inputs for attributes generated from the prepared element bytes.
    #[cfg(feature = "provenance")]
    pub(crate) provenance: Option<Provenance>,
}

/// A creation request with providers and committed attribute paths available for file placement.
pub(crate) struct WholeFileDataset {
    pub(crate) metadata: DatasetMetadata,
    pub(crate) storage: PreparedStorage,
    /// Attribute indices paired with committed datatype paths awaiting address resolution.
    pub(crate) committed_datatype_paths: Vec<(usize, ObjectPathBuf)>,
}

/// A creation request with inline datatypes and storage without providers.
#[cfg(feature = "std")]
pub(crate) struct InPlaceDataset {
    pub(crate) metadata: DatasetMetadata,
    pub(crate) storage: PreparedElements,
}

/// Replacement values without creation options, pending compatibility with an existing dataset.
#[cfg(feature = "std")]
pub(crate) struct ValueOverwrite {
    pub(crate) dt: Datatype,
    pub(crate) ds: Dataspace,
    bytes: PreparedBytes,
}

#[cfg(feature = "std")]
impl ValueOverwrite {
    /// Takes the replacement buffer and deferred heap staging, leaving an empty buffer and `None`.
    pub(crate) fn take_payload(&mut self) -> (Vec<u8>, Option<VlStringStaging>) {
        (
            core::mem::take(&mut self.bytes.raw),
            self.bytes.vl_string_staging.take(),
        )
    }

    /// Returns replacement element bytes before deferred heap addresses are assigned.
    pub(crate) fn raw(&self) -> &[u8] {
        &self.bytes.raw
    }

    /// Returns the variable-length heap collections and reference offsets, if present.
    pub(crate) fn vl_string_staging(&self) -> Option<&VlStringStaging> {
        self.bytes.vl_string_staging.as_ref()
    }
}

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
    #[cfg(feature = "std")]
    use crate::type_builders::DatasetBuilder;
    use crate::type_builders::{ProducedPayload, RawChunkPayload, VlStringStaging};
    #[cfg(feature = "std")]
    use crate::{Error, FillTime};

    #[cfg(feature = "std")]
    use super::PreparedDataset;
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

    #[cfg(feature = "std")]
    #[rstest]
    #[case::default_policy(FillTime::IfSet)]
    #[case::allocate(FillTime::Alloc)]
    #[case::never(FillTime::Never)]
    fn overwrites_reject_explicit_fill_time(#[case] fill_time: FillTime) {
        let mut builder = DatasetBuilder::new("values");
        builder.with_i32_data(&[1]).with_fill_time(fill_time);
        let result = PreparedDataset::parse(builder).unwrap().into_overwrite();
        let Err(Error::EditUnsupported(reason)) = result else {
            panic!("expected an unsupported overwrite option");
        };
        assert_eq!(
            reason,
            "write_dataset overwrites values only; it cannot change the fill time (set it when the dataset is created)"
        );
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
