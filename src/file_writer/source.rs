//! File layout and emission from an indexed source of object metadata.
//!
//! The writer reconstructs individual objects during planning and emission. The retained plan
//! contains allocations, object-use counts, and serialized file-space managers.

use core::num::NonZeroU64;

use std::borrow::Cow;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;

use hdf5_pure_core::__private::SuperblockFields;
use hdf5_pure_format::__private::DenseAttrCreationOrder;
use hdf5_pure_format::__private::DenseAttrFixup;
use hdf5_pure_format::__private::PreparedDenseAttrs;
use hdf5_pure_space::__private::AllocationClass;

use super::layout;
use super::layout::Allocation;
use super::layout::ObjectAllocation;
use super::layout::ObjectMeasure;
use super::preparation::PreparedAttributes;
use super::preparation::PreparedDataset;
use super::preparation::VlPatch;
use crate::FileSpaceSettings;
use crate::SuperblockStatus;
use crate::SuperblockStatusFlags;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::attribute::AttributeMessage;
use crate::chunked_write::ByteSink;
use crate::chunked_write::CompressedChunkSet;
use crate::chunked_write::EmptyChunkIndex;
use crate::convert::Narrow;
use crate::datatype::Datatype;
use crate::error::Error;
use crate::error::FormatError;
use crate::file_allocation::FileAllocation;
use crate::file_allocation::FileManager;
use crate::file_emission::EmissionError;
use crate::file_space_info;
use crate::file_space_info::DEFAULT_THRESHOLD;
use crate::file_space_info::FileSpaceStrategy;
use crate::libver::LibVer;
use crate::link_info::LinkInfoMessage;
use crate::link_message::LinkMessage;
use crate::object_path::ObjectPathBuf;
use crate::shared_message::DatatypeLocation;
use crate::type_builders;
use crate::type_builders::AttributeSpecs;
use crate::type_builders::DatasetBuilder;
use crate::type_builders::ObjectRefTarget;
use crate::writer::LazyFile;
use crate::writer::WriteSink;

/// An indexed source of reproducible object metadata and hard links.
///
/// The final index identifies the root group. Indices, links, and object contents must remain
/// consistent across calls during planning and emission.
pub(crate) trait ObjectSource {
    /// Returns the number of objects, including the root group.
    fn len(&self) -> usize;

    /// Reconstructs the object at `index` with unresolved destination addresses.
    ///
    /// # Errors
    ///
    /// Propagates source reading and object preparation errors.
    fn load(&self, index: usize) -> Result<Object, Error>;

    /// Releases source state retained for initial object measurement.
    ///
    /// Called after metadata reservation and before final layout checks and emission.
    /// The default implementation is empty.
    fn finish_planning(&self) {}

    /// Returns each hard link of the object at `index`, with its target's index.
    ///
    /// Link names may borrow from the source.
    ///
    /// # Errors
    ///
    /// Propagates source reading and object index errors.
    fn links(
        &self,
        index: usize,
    ) -> Result<impl ExactSizeIterator<Item = (Cow<'_, str>, usize)>, Error>;

    /// Returns the index of the object at `path` under the root group.
    ///
    /// # Errors
    ///
    /// Propagates path resolution errors.
    fn resolve(&self, path: &ObjectPathBuf) -> Result<usize, Error>;
}

/// A prepared object and its variable-length attribute heap collections.
///
/// Dataset element heap collections are held in prepared storage.
pub(crate) struct Object {
    kind: Kind,
    heaps: Vec<VlPatch>,
    /// Exclusive collection end indices for the attribute heap patches in `heaps`.
    heap_ends: Vec<usize>,
}

impl Object {
    /// Prepares dataset metadata, attributes, and its storage source.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`PreparedDataset::prepare`].
    pub(crate) fn dataset(builder: DatasetBuilder) -> Result<Self, FormatError> {
        let (dataset, heaps) = PreparedDataset::prepare(builder)?;
        Self::new(Kind::Dataset(Box::new(dataset)), heaps)
    }

    /// Prepares a group's attributes without its hard links.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`PreparedAttributes::new`].
    pub(crate) fn group(attributes: &AttributeSpecs) -> Result<Self, FormatError> {
        let PreparedAttributes {
            messages,
            committed_datatype_paths,
            heaps,
        } = PreparedAttributes::new(attributes)?;
        Self::new(
            Kind::Group {
                attributes: messages,
                paths: committed_datatype_paths,
            },
            heaps,
        )
    }

    /// Prepares a committed datatype with a nonzero element size.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Datatype::element_size`].
    pub(crate) fn committed(datatype: Datatype) -> Result<Self, FormatError> {
        datatype.element_size()?;
        Self::new(Kind::Committed(datatype), Vec::new())
    }

    /// Constructs an object with an index of its attribute heap collections.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the total collection count exceeds [`usize::MAX`].
    fn new(kind: Kind, heaps: Vec<VlPatch>) -> Result<Self, FormatError> {
        let mut end = 0_usize;
        let heap_ends = heaps
            .iter()
            .map(|patch| {
                end = end
                    .checked_add(patch.collections.len())
                    .ok_or_else(inconsistent)?;
                Ok(end)
            })
            .collect::<Result<_, FormatError>>()?;
        Ok(Self {
            kind,
            heaps,
            heap_ends,
        })
    }

    /// Adds dataset and attribute uses of committed datatypes to `references`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::CommittedDatatypeMismatch`] if a use differs from its committed
    /// datatype. Returns [`FormatError::Internal`] if a target is not a committed datatype or its
    /// index is outside `references`. Propagates source loading and path resolution errors.
    fn count_uses(&self, source: &impl ObjectSource, references: &mut [u32]) -> Result<(), Error> {
        let mut use_type = |path: &ObjectPathBuf, datatype: &Datatype| -> Result<(), Error> {
            let target = source.resolve(path)?;
            let Kind::Committed(committed) = source.load(target)?.kind else {
                return Err(inconsistent().into());
            };
            if hdf5_pure_format::__private::serialize_datatype(&committed)
                != hdf5_pure_format::__private::serialize_datatype(datatype)
            {
                return Err(FormatError::CommittedDatatypeMismatch {
                    path: path.to_string(),
                    user: "a repacked object".into(),
                }
                .into());
            }
            let count = references.get_mut(target).ok_or_else(inconsistent)?;
            *count = count.saturating_add(1);
            Ok(())
        };
        for (index, path) in self.paths() {
            let attribute = self.attributes().get(*index).ok_or_else(inconsistent)?;
            use_type(path, &attribute.datatype)?;
        }
        if let Kind::Dataset(dataset) = &self.kind {
            if let Some(path) = &dataset.committed_datatype_path {
                use_type(path, &dataset.dt)?;
            }
        }
        Ok(())
    }

    /// Measures an object's metadata and data regions after assigning heap addresses.
    ///
    /// A committed datatype's record has a provisional header length of zero. The writer measures
    /// its header after counting all uses.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if heap allocations differ from the prepared collections.
    /// Propagates source link, heap patching, chunk preparation, and serialization errors.
    fn measure(
        mut self,
        source: &impl ObjectSource,
        index: usize,
        libver: LibVer,
        heaps: Vec<Allocation>,
    ) -> Result<Record, Error> {
        self.patch_heaps(&heaps)?;
        let dense = self.dense()?;
        let info = dense
            .as_ref()
            .map(|dense| dense.attr_info_message(StoredAddress::ZERO))
            .transpose()?;
        let (header, data, chunked, links, empty_index, dense_links) = match &self.kind {
            Kind::Dataset(dataset) => {
                let chunks = self.chunks()?;
                let measured = dataset.measure(
                    StoredAddress::ZERO,
                    chunks.as_ref(),
                    info.as_deref(),
                    libver,
                )?;
                (
                    measured.header_len as u64,
                    Some(Allocation {
                        address: StoredAddress::ZERO,
                        length: measured.data_len,
                    }),
                    dataset.is_chunked(),
                    None,
                    chunks.as_ref().and_then(CompressedChunkSet::empty_index),
                    None,
                )
            }
            Kind::Group { attributes, .. } => {
                let links = links_at(source, index, None)?;
                let plan = super::dense_links_plan(&links)?;
                let link_info = plan
                    .as_ref()
                    .map(|plan| plan.link_info(StoredAddress::ZERO))
                    .transpose()
                    .map_err(super::dense_link_error)?;
                let header =
                    super::build_group_oh(&links, attributes, info.as_deref(), link_info.as_ref())?;
                (
                    header.len() as u64,
                    None,
                    false,
                    plan.as_ref().map(|plan| Allocation {
                        address: StoredAddress::ZERO,
                        length: plan.serialized_len(),
                    }),
                    None,
                    link_info.as_ref().map(DenseLinks::new).transpose()?,
                )
            }
            Kind::Committed(_) => (0, None, false, None, None, None),
        };
        let preflight = !heaps.is_empty()
            || dense.is_some()
            || !self.paths().is_empty()
            || matches!(&self.kind, Kind::Dataset(dataset) if dataset.storage.reference_targets().is_some() || dataset.committed_datatype_path.is_some());
        Ok(Record {
            metadata: ObjectAllocation {
                header: Allocation {
                    address: StoredAddress::ZERO,
                    length: header,
                },
                attributes: dense.map(|dense| Allocation {
                    address: StoredAddress::ZERO,
                    length: dense.blob_len(),
                }),
                links,
            },
            data,
            chunked,
            empty_index,
            dense_links,
            heaps,
            references: 1,
            preflight,
        })
    }

    fn attributes(&self) -> &[AttributeMessage] {
        match &self.kind {
            Kind::Dataset(dataset) => &dataset.attrs,
            Kind::Group { attributes, .. } => attributes,
            Kind::Committed(_) => &[],
        }
    }

    fn attributes_mut(&mut self) -> &mut [AttributeMessage] {
        match &mut self.kind {
            Kind::Dataset(dataset) => &mut dataset.attrs,
            Kind::Group { attributes, .. } => attributes,
            Kind::Committed(_) => &mut [],
        }
    }

    /// Returns attribute indices paired with committed datatype paths.
    fn paths(&self) -> &[(usize, ObjectPathBuf)] {
        match &self.kind {
            Kind::Dataset(dataset) => &dataset.attr_committed_datatype_paths,
            Kind::Group { paths, .. } => paths,
            Kind::Committed(_) => &[],
        }
    }

    /// Returns attribute heap collections followed by dataset element heap collections.
    fn collections(&self) -> impl Iterator<Item = &Vec<u8>> {
        let Self {
            kind,
            heaps,
            heap_ends: _,
        } = self;
        heaps.iter().flat_map(|patch| &patch.collections).chain(
            match kind {
                Kind::Dataset(dataset) => dataset
                    .storage
                    .vl_string_staging()
                    .map(|staging| &staging.collections),
                _ => None,
            }
            .into_iter()
            .flatten(),
        )
    }

    /// Returns the heap collection at `index` in [`Self::collections`] order.
    ///
    /// Returns [`None`] if `index` is outside that sequence.
    fn collection(&self, index: usize) -> Option<&Vec<u8>> {
        let Self {
            kind,
            heaps,
            heap_ends,
        } = self;
        let patch = heap_ends.partition_point(|&end| end <= index);
        if let Some(heap) = heaps.get(patch) {
            let start = if patch == 0 {
                0
            } else {
                *heap_ends.get(patch - 1)?
            };
            return heap.collections.get(index.checked_sub(start)?);
        }
        let Kind::Dataset(dataset) = kind else {
            return None;
        };
        dataset
            .storage
            .vl_string_staging()?
            .collections
            .get(index.checked_sub(heap_ends.last().copied().unwrap_or(0))?)
    }

    /// Assigns collection addresses to variable-length attribute and dataset references.
    ///
    /// `spans` has the order of [`Self::collections`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if collection counts, lengths, or attribute indices differ
    /// from the prepared metadata. Propagates errors accessing dataset element bytes.
    fn patch_heaps(&mut self, spans: &[Allocation]) -> Result<(), FormatError> {
        let Self {
            kind,
            heaps,
            heap_ends: _,
        } = self;
        let addresses: Vec<u64> = spans.iter().map(|span| span.address.get()).collect();
        let mut cursor = 0;
        for patch in heaps.iter() {
            let end = cursor + patch.collections.len();
            let addresses = addresses.get(cursor..end).ok_or_else(inconsistent)?;
            match &mut *kind {
                Kind::Dataset(dataset) => type_builders::patch_vl_refs(
                    &mut dataset
                        .attrs
                        .get_mut(patch.attr_index)
                        .ok_or_else(inconsistent)?
                        .raw_data,
                    addresses,
                ),
                Kind::Group { attributes, .. } => type_builders::patch_vl_refs(
                    &mut attributes
                        .get_mut(patch.attr_index)
                        .ok_or_else(inconsistent)?
                        .raw_data,
                    addresses,
                ),
                Kind::Committed(_) => return Err(inconsistent()),
            }
            cursor = end;
        }
        if let Kind::Dataset(dataset) = &mut *kind {
            if let Some(staging) = dataset.storage.take_vl_string_staging() {
                let end = cursor + staging.collections.len();
                type_builders::patch_vl_refs_masked(
                    dataset.storage.raw_mut()?,
                    &staging.patch_offsets,
                    addresses.get(cursor..end).ok_or_else(inconsistent)?,
                );
                dataset.storage.restore_vl_string_staging(staging)?;
                cursor = end;
            }
        }
        if cursor != spans.len()
            || self
                .collections()
                .zip(spans)
                .any(|(bytes, span)| bytes.len() as u64 != span.length)
        {
            return Err(inconsistent());
        }
        Ok(())
    }

    /// Prepares dense attribute storage when the attributes require it.
    ///
    /// # Errors
    ///
    /// Propagates attribute sizing and dense storage preparation errors.
    fn dense(&self) -> Result<Option<PreparedDenseAttrs>, FormatError> {
        let attributes = self.attributes();
        if !super::needs_dense_attrs(attributes)? {
            return Ok(None);
        }
        let fixups = self
            .heaps
            .iter()
            .map(|patch| DenseAttrFixup::GlobalHeap {
                attribute: patch.attr_index,
            })
            .chain(
                attributes
                    .iter()
                    .enumerate()
                    .filter_map(|(attribute, attr)| {
                        attr.datatype_location
                            .is_committed()
                            .then_some(DenseAttrFixup::CommittedDatatype { attribute })
                    }),
            );
        PreparedDenseAttrs::new(attributes, DenseAttrCreationOrder::Untracked, fixups).map(Some)
    }

    /// Encodes chunks for a chunked dataset whose storage is not verbatim.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`PreparedDataset::compress_chunks`].
    fn chunks(&self) -> Result<Option<CompressedChunkSet>, FormatError> {
        match &self.kind {
            Kind::Dataset(dataset)
                if dataset.is_chunked() && dataset.storage.chunks().is_none() =>
            {
                dataset.compress_chunks().map(Some)
            }
            _ => Ok(None),
        }
    }
}

/// The prepared content of a committed datatype, dataset, or group.
enum Kind {
    Committed(Datatype),
    Dataset(Box<PreparedDataset>),
    Group {
        attributes: Vec<AttributeMessage>,
        /// Indices in `attributes` paired with the paths of their committed datatypes.
        paths: Vec<(usize, ObjectPathBuf)>,
    },
}

/// Stored allocations and file metadata for emission from a retained source.
///
/// Object metadata is reconstructed from `source` for layout checks and emission. Each record
/// holds allocation spans and counts. Serialized file-space managers are retained.
pub(crate) struct Plan<S> {
    source: S,
    records: Vec<Record>,
    libver: LibVer,
    extension: Option<(Allocation, Vec<u8>)>,
    managers: Vec<FileManager>,
    end: u64,
}

impl<S: ObjectSource> Plan<S> {
    /// Plans object allocations, resolves references, and checks metadata before output.
    ///
    /// Uses `inherited` when `bounds` is absent, upgrading the format for content that requires
    /// it. Each source object is reconstructed for measurement and, where needed, final checks.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::LibverTooOldForContent`] if explicit bounds cannot represent the
    /// content. Propagates source, preparation, allocation, and metadata serialization errors.
    pub(crate) fn new(
        source: S,
        settings: Option<FileSpaceSettings>,
        bounds: Option<(LibVer, LibVer)>,
        inherited: LibVer,
    ) -> Result<Self, Error> {
        let mut libver = LibVer::resolve_writable(bounds.or(Some((LibVer::Earliest, inherited))))?;
        let mut latest = settings.is_some();
        let threshold = match settings {
            None => Some(DEFAULT_THRESHOLD),
            Some(settings) if settings.strategy() == FileSpaceStrategy::FsmAggr => {
                Some(settings.threshold())
            }
            _ => None,
        };
        let paged = settings.filter(|settings| settings.strategy() == FileSpaceStrategy::Page);
        let aggregated = paged.is_none()
            && (threshold.is_some()
                || settings.is_some_and(|settings| settings.strategy() == FileSpaceStrategy::Aggr));
        let mut allocation = FileAllocation::new(
            StoredAddress::new(super::SUPERBLOCK_SIZE as u64),
            aggregated,
            threshold,
        );
        let mut collections = paged.map(|settings| {
            layout::PagedCollections::new(super::SUPERBLOCK_SIZE as u64, settings.page_size())
        });
        let mut records = Vec::with_capacity(source.len());
        let mut references = vec![1u32; source.len()];
        for index in 0..source.len() {
            let object = source.load(index)?;
            object.count_uses(&source, &mut references)?;
            if let Kind::Dataset(dataset) = &object.kind {
                latest |= dataset.is_chunked();
            }
            let heaps = object
                .collections()
                .map(|bytes| match &mut collections {
                    Some(collections) => collections.reserve(bytes.len() as u64),
                    None => Ok(Allocation {
                        address: allocation.reserve(bytes.len() as u64, AllocationClass::Raw)?,
                        length: bytes.len() as u64,
                    }),
                })
                .collect::<Result<Vec<_>, FormatError>>()?;
            records.push(object.measure(&source, index, libver, heaps)?);
        }
        if latest && bounds.is_none() {
            libver = LibVer::WRITER_DEFAULT;
        }
        if latest && libver < LibVer::V110 {
            return Err(FormatError::LibverTooOldForContent {
                content: if settings.is_some() {
                    "a file-space setting"
                } else {
                    "a chunked, filtered, or resizable dataset"
                },
                needs: LibVer::V110.name(),
                writing: libver.name(),
            }
            .into());
        }
        if let Some(collections) = &collections {
            allocation = FileAllocation::new(StoredAddress::new(collections.end), false, None);
        }
        reserve_metadata(&source, &mut records, &references, &mut allocation)?;
        drop(references);
        source.finish_planning();
        let FileLayout {
            extension,
            end,
            managers,
        } = FileLayout::finish(allocation, collections, settings, &mut records)?;
        let plan = Self {
            source,
            records,
            libver,
            extension,
            managers,
            end,
        };
        plan.preflight()?;
        Ok(plan)
    }

    /// Serializes the planned file into memory.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::finish_to`].
    #[cfg(test)]
    pub(crate) fn finish(self) -> Result<Vec<u8>, Error> {
        let mut bytes = Vec::new();
        self.finish_to(&mut bytes)?;
        Ok(bytes)
    }

    /// Writes the plan to `path`, creating the file on the first output write.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::finish_to`]. An emission failure can leave a partial file.
    pub(crate) fn write(self, path: impl AsRef<Path>) -> Result<(), Error> {
        self.finish_to(LazyFile {
            path: path.as_ref().to_path_buf(),
            file: None,
        })
    }

    /// Writes allocations in address order while reconstructing one source object at a time.
    ///
    /// The working object is retained across adjacent allocations of that object. Chunk providers
    /// supply verbatim payloads during data emission. The writer encodes chunks when emitting
    /// a header or data region.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if reconstructed output differs from its planned spans.
    /// Propagates source, reference resolution, serialization, provider, and output I/O errors.
    pub(crate) fn finish_to(self, writer: impl Write) -> Result<(), Error> {
        let Self {
            source: _,
            records,
            libver,
            extension,
            managers,
            end,
        } = &self;
        let mut sink = WriteSink::new(BufWriter::new(writer));
        let superblock = SuperblockFields {
            status: SuperblockStatus::new(
                super::superblock_version(*libver),
                SuperblockStatusFlags::new(),
            )?,
            offset_size: super::OFFSET_WIDTH,
            length_size: super::LENGTH_WIDTH,
            base_address: BaseAddress::ZERO,
            eof_address: *end,
            root_group_address: records
                .last()
                .ok_or_else(inconsistent)?
                .metadata
                .header
                .address,
            group_leaf_node_k: None,
            group_internal_node_k: None,
            indexed_storage_internal_node_k: None,
            free_space_address: None,
            driver_info_address: None,
            superblock_extension_address: extension.as_ref().map(|(span, _)| span.address),
            checksum: None,
        }
        .build();
        sink.put(&hdf5_pure_format::__private::serialize_superblock(
            &superblock,
        )?)?;
        let mut spans = Vec::new();
        for (index, record) in records.iter().enumerate() {
            spans.push((record.metadata.header, Part::Header(index)));
            if let Some(span) = record.metadata.attributes {
                spans.push((span, Part::Attributes(index)));
            }
            if let Some(span) = record.metadata.links {
                spans.push((span, Part::Links(index)));
            }
            for (collection, &span) in record.heaps.iter().enumerate() {
                spans.push((span, Part::Heap(index, collection)));
            }
            if let Some(span) = record.data.filter(|span| span.length != 0) {
                spans.push((
                    span,
                    record
                        .empty_index
                        .map_or(Part::Data(index), Part::EmptyIndex),
                ));
            }
        }
        if let Some((span, _)) = extension.as_ref() {
            spans.push((*span, Part::Extension));
        }
        for (index, manager) in managers.iter().enumerate() {
            spans.push((
                Allocation {
                    address: manager.header_address,
                    length: manager.header.len() as u64,
                },
                Part::ManagerHeader(index),
            ));
            spans.push((
                Allocation {
                    address: manager.section_list_address,
                    length: manager.section_list.len() as u64,
                },
                Part::ManagerSections(index),
            ));
        }
        spans.sort_unstable_by_key(|(span, _)| span.address);
        let mut working = None;
        for (span, part) in spans {
            let start = BaseAddress::ZERO.resolve(span.address)?.get();
            let end = start
                .checked_add(span.length)
                .ok_or(FormatError::OffsetOverflow {
                    offset: start,
                    length: span.length,
                })?;
            pad(&mut sink, start)?;
            if let Some(index) = part.object() {
                if !working
                    .as_ref()
                    .is_some_and(|(current, _, _)| *current == index)
                {
                    drop(working.take());
                    let object = self.load(index)?;
                    working = Some((index, object, None));
                }
            }
            match part {
                Part::EmptyIndex(index) => sink.put(&index.build_at(span.address)?)?,
                Part::Links(index) => sink.put(&self.links(index, span)?)?,
                Part::Extension => sink.put(&extension.as_ref().ok_or_else(inconsistent)?.1)?,
                Part::ManagerHeader(index) => sink.put(&managers[index].header)?,
                Part::ManagerSections(index) => sink.put(&managers[index].section_list)?,
                part => {
                    let (index, object, chunks) = working.as_mut().ok_or_else(inconsistent)?;
                    if chunks.is_none() && matches!(part, Part::Header(_) | Part::Data(_)) {
                        *chunks = object.chunks()?;
                    }
                    match part {
                        Part::Header(_) => {
                            sink.put(&self.header(*index, object, chunks.as_ref())?)?
                        }
                        Part::Attributes(_) => sink.put(&self.attributes(object, span)?)?,
                        Part::Heap(_, collection) => {
                            sink.put(object.collection(collection).ok_or_else(inconsistent)?)?
                        }
                        Part::Data(_) => {
                            let Kind::Dataset(dataset) = &mut object.kind else {
                                return Err(inconsistent().into());
                            };
                            if dataset.is_chunked() {
                                dataset
                                    .build_chunked(span.address, chunks.as_ref())?
                                    .data
                                    .emit(&mut sink, &dataset.storage)
                                    .map_err(emission_error)?;
                            } else {
                                dataset
                                    .contiguous_data()
                                    .emit(&mut sink, &dataset.storage)
                                    .map_err(emission_error)?;
                            }
                        }
                        _ => return Err(inconsistent().into()),
                    }
                }
            }
            if sink.position() != end {
                return Err(inconsistent().into());
            }
        }
        pad(&mut sink, *end)?;
        sink.into_inner().flush()?;
        Ok(())
    }

    /// Checks relocated metadata and lengths for records that require destination addresses.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if emitted lengths would differ from the reservations.
    /// Propagates source loading, relocation, and serialization errors.
    fn preflight(&self) -> Result<(), Error> {
        for index in 0..self.records.len() {
            if !self.records[index].preflight {
                continue;
            }
            let object = self.load(index)?;
            let chunks = object.chunks()?;
            let record = &self.records[index];
            check_length(
                &self.header(index, &object, chunks.as_ref())?,
                record.metadata.header,
            )?;
            if let Some(span) = record.metadata.attributes {
                check_length(&self.attributes(&object, span)?, span)?;
            }
            if let (Kind::Dataset(dataset), Some(span)) = (&object.kind, record.data) {
                let length = if record.chunked {
                    dataset
                        .build_chunked(span.address, chunks.as_ref())?
                        .data
                        .len()
                } else {
                    dataset.contiguous_len()
                };
                if length != span.length {
                    return Err(inconsistent().into());
                }
            }
        }
        Ok(())
    }

    /// Reconstructs an object and resolves its heap, committed datatype, and object references.
    ///
    /// # Errors
    ///
    /// Propagates source loading, reference resolution, heap patching, and content hash errors.
    fn load(&self, index: usize) -> Result<Object, Error> {
        let mut object = self.source.load(index)?;
        let record = self.records.get(index).ok_or_else(inconsistent)?;
        object.patch_heaps(&record.heaps)?;
        let locations = object
            .paths()
            .iter()
            .map(|(index, path)| self.address(path).map(|address| (*index, address)))
            .collect::<Result<Vec<_>, Error>>()?;
        for (index, location) in locations {
            object
                .attributes_mut()
                .get_mut(index)
                .ok_or_else(inconsistent)?
                .datatype_location = DatatypeLocation::Committed(location);
        }
        if let Kind::Dataset(dataset) = &mut object.kind {
            if let Some(path) = &dataset.committed_datatype_path {
                dataset.dt_location = DatatypeLocation::Committed(self.address(path)?);
            }
            if let Some(patches) = dataset.storage.take_reference_targets() {
                for patch in patches {
                    let address = match patch.target {
                        ObjectRefTarget::Path(path) => self.address(&path)?,
                        ObjectRefTarget::Raw(address) => address,
                    };
                    type_builders::write_reference_address(
                        dataset.storage.raw_mut()?,
                        patch.byte_offset,
                        address,
                    );
                }
            }
            #[cfg(feature = "provenance")]
            if let Some(hash) = dataset.pending_hash.take() {
                hash.complete(&mut dataset.attrs, dataset.storage.raw())?;
            }
        }
        Ok(object)
    }

    /// Returns the destination header address for a source object path.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if a resolved index is outside the plan. Propagates
    /// path resolution errors.
    fn address(&self, path: &ObjectPathBuf) -> Result<StoredAddress, Error> {
        Ok(self
            .records
            .get(self.source.resolve(path)?)
            .ok_or_else(inconsistent)?
            .metadata
            .header
            .address)
    }

    /// Serializes an object's header using its planned addresses and committed-type use count.
    ///
    /// # Errors
    ///
    /// Propagates source link, chunk measurement, dense metadata, and header serialization errors.
    fn header(
        &self,
        index: usize,
        object: &Object,
        chunks: Option<&CompressedChunkSet>,
    ) -> Result<Vec<u8>, Error> {
        let record = &self.records[index];
        let info = object
            .dense()?
            .map(|dense| {
                dense
                    .attr_info_message(record.metadata.attributes.ok_or_else(inconsistent)?.address)
            })
            .transpose()?;
        Ok(match &object.kind {
            Kind::Dataset(dataset) => {
                let address = record
                    .data
                    .map_or(super::UNDEF_ADDRESS, |span| span.address);
                let measured = dataset
                    .is_chunked()
                    .then(|| dataset.measure_chunked(address, chunks))
                    .transpose()?;
                let messages =
                    measured.map(|measured| (measured.layout_message, measured.pipeline_message));
                dataset.build_header(address, messages.as_ref(), info.as_deref(), self.libver)?
            }
            Kind::Group { attributes, .. } => {
                let link_info = record.link_info()?;
                let links = if link_info.is_some() {
                    Vec::new()
                } else {
                    links_at(&self.source, index, Some(&self.records))?
                };
                super::build_group_oh(&links, attributes, info.as_deref(), link_info.as_ref())?
            }
            Kind::Committed(datatype) => {
                super::build_committed_datatype_oh(datatype, record.references)?
            }
        })
    }

    /// Serializes dense attribute storage at its reserved address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the object has no dense attribute storage. Propagates
    /// dense attribute preparation and serialization errors.
    fn attributes(&self, object: &Object, span: Allocation) -> Result<Vec<u8>, FormatError> {
        let prepared = object.dense()?.ok_or_else(inconsistent)?;
        Ok(super::finalize_dense_attrs(prepared, object.attributes())?
            .build(span.address)?
            .blob)
    }

    /// Serializes a group's dense links with destination object addresses.
    ///
    /// # Errors
    ///
    /// Propagates source link, allocation length conversion, and dense link serialization errors.
    fn links(&self, index: usize, span: Allocation) -> Result<Vec<u8>, Error> {
        Ok(super::dense_links_image(
            &links_at(&self.source, index, Some(&self.records))?,
            (span.address, span.length.to_usize()?),
        )?
        .bytes)
    }
}

/// The name-index offset within untracked dense link storage.
#[derive(Clone, Copy)]
struct DenseLinks {
    /// Relative to the stored address of the fractal heap.
    name_index_offset: NonZeroU64,
}

impl DenseLinks {
    /// Records relative name-index geometry measured with the heap at address zero.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if creation-order metadata is present, the heap
    /// address is absent or nonzero, or the name-index address is absent or zero.
    fn new(info: &LinkInfoMessage) -> Result<Self, FormatError> {
        let LinkInfoMessage {
            max_creation_order,
            fractal_heap_address,
            btree_name_index_address,
            btree_creation_order_address,
        } = info;
        if max_creation_order.is_some()
            || *fractal_heap_address != Some(StoredAddress::ZERO)
            || btree_creation_order_address.is_some()
        {
            return Err(inconsistent());
        }
        Ok(Self {
            name_index_offset: NonZeroU64::new(
                btree_name_index_address.ok_or_else(inconsistent)?.get(),
            )
            .ok_or_else(inconsistent)?,
        })
    }

    /// Returns stored heap and name-index addresses for `span`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if `span` starts at the undefined address,
    /// [`FormatError::OffsetOverflow`] if its end exceeds [`u64::MAX`], or
    /// [`FormatError::Internal`] if the name-index offset is outside the allocation.
    fn link_info(self, span: Allocation) -> Result<LinkInfoMessage, FormatError> {
        let Self { name_index_offset } = self;
        let Allocation { address, length } = span;
        if address == super::UNDEF_ADDRESS {
            return Err(FormatError::AddressTooLarge {
                address: address.get(),
                offset_size: super::OFFSET_WIDTH.get(),
            });
        }
        address
            .get()
            .checked_add(length)
            .ok_or(FormatError::OffsetOverflow {
                offset: address.get(),
                length,
            })?;
        if name_index_offset.get() >= length {
            return Err(inconsistent());
        }
        let name_index = address.get().checked_add(name_index_offset.get()).ok_or(
            FormatError::OffsetOverflow {
                offset: address.get(),
                length: name_index_offset.get(),
            },
        )?;
        Ok(LinkInfoMessage {
            max_creation_order: None,
            fractal_heap_address: Some(address),
            btree_name_index_address: Some(StoredAddress::new(name_index)),
            btree_creation_order_address: None,
        })
    }
}

/// Allocations, storage kind, and relocation state for one source object.
struct Record {
    metadata: ObjectAllocation,
    data: Option<Allocation>,
    chunked: bool,
    /// Geometry for an array index with no stored chunks.
    empty_index: Option<EmptyChunkIndex>,
    /// Relative name-index geometry for the dense links in `metadata`.
    dense_links: Option<DenseLinks>,
    heaps: Vec<Allocation>,
    references: u32,
    /// Whether heaps, dense attributes, or references require final metadata checks.
    preflight: bool,
}

impl Record {
    /// Returns dense link addresses from the reserved span and relative geometry.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if only the span or geometry is present. Propagates
    /// errors from [`DenseLinks::link_info`].
    fn link_info(&self) -> Result<Option<LinkInfoMessage>, FormatError> {
        match (self.dense_links, self.metadata.links) {
            (Some(info), Some(span)) => info.link_info(span).map(Some),
            (None, None) => Ok(None),
            _ => Err(inconsistent()),
        }
    }

    fn measure(&self) -> layout::DataMeasure {
        layout::DataMeasure::new(
            self.data.map_or(0, |span| span.length),
            if self.chunked {
                layout::DataKind::Chunked
            } else {
                layout::DataKind::Contiguous
            },
        )
    }
}

/// The source of bytes for one allocation in the ordered output.
enum Part {
    Attributes(usize),
    Data(usize),
    /// An array index emitted from scalar geometry.
    EmptyIndex(EmptyChunkIndex),
    Extension,
    Header(usize),
    Heap(usize, usize),
    Links(usize),
    ManagerHeader(usize),
    ManagerSections(usize),
}

impl Part {
    fn object(&self) -> Option<usize> {
        match *self {
            Self::Attributes(index)
            | Self::Data(index)
            | Self::Header(index)
            | Self::Heap(index, _) => Some(index),
            Self::EmptyIndex(_)
            | Self::Links(_)
            | Self::Extension
            | Self::ManagerHeader(_)
            | Self::ManagerSections(_) => None,
        }
    }
}

/// Reserves root, group, committed datatype, and dataset metadata in that order.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if a committed datatype record has another object kind.
/// Propagates source loading, header serialization, and allocation errors.
fn reserve_metadata(
    source: &impl ObjectSource,
    records: &mut [Record],
    references: &[u32],
    allocation: &mut FileAllocation,
) -> Result<(), Error> {
    let root = records.len().checked_sub(1).ok_or_else(inconsistent)?;
    let mut order: Vec<_> = (0..records.len()).collect();
    order.sort_by_key(|&index| {
        let record = &records[index];
        if index == root {
            0
        } else if record.data.is_some() {
            3
        } else if record.metadata.header.length == 0 {
            2
        } else {
            1
        }
    });
    for index in order {
        let record = &mut records[index];
        record.references = references[index];
        if record.metadata.header.length == 0 {
            let Kind::Committed(datatype) = source.load(index)?.kind else {
                return Err(inconsistent().into());
            };
            record.metadata.header.length =
                super::build_committed_datatype_oh(&datatype, record.references)?.len() as u64;
        }
        let ObjectAllocation {
            header,
            attributes,
            links,
        } = record.metadata;
        record.metadata = ObjectMeasure {
            header: header.length,
            attributes: attributes.map(|span| span.length),
            links: links.map(|span| span.length),
        }
        .reserve(allocation)?;
        if let Some(info) = record.dense_links {
            info.link_info(record.metadata.links.ok_or_else(inconsistent)?)?;
        }
    }
    Ok(())
}

/// The superblock extension, serialized managers, and stored end of allocation.
struct FileLayout {
    extension: Option<(Allocation, Vec<u8>)>,
    end: u64,
    managers: Vec<FileManager>,
}

impl FileLayout {
    /// Places dataset regions and serializes file-space settings and managers.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if extension lengths differ from their reservations.
    /// Propagates classic or PAGE allocation and file-space serialization errors.
    fn finish(
        mut allocation: FileAllocation,
        collections: Option<layout::PagedCollections>,
        settings: Option<FileSpaceSettings>,
        records: &mut [Record],
    ) -> Result<Self, Error> {
        let extension = settings
            .map(|settings| {
                super::file_space_extension_oh(&super::file_space_placeholder(settings))
            })
            .transpose()?;
        let (extension_span, end, managers, extension_info) = if let Some(settings) =
            settings.filter(|settings| settings.strategy() == FileSpaceStrategy::Page)
        {
            let measures: Vec<_> = records.iter().map(Record::measure).collect();
            let layout::PagedLayout {
                extension,
                data: _,
                small: _,
                large: _,
                slots,
                managers,
                metadata_end: _,
                raw_start: _,
                small_end: _,
                end,
            } = layout::PagedLayout::new(
                settings.page_size(),
                settings.persist(),
                allocation.end().get(),
                extension.as_ref().map_or(0, |bytes| bytes.len() as u64),
                measures.iter().copied(),
                collections.ok_or_else(inconsistent)?.sections,
                |index, placed| {
                    records[index].data = placed;
                    Ok(())
                },
            )?;
            let info = settings.persist().then(|| {
                file_space_info::persistent_managers(
                    FileSpaceStrategy::Page,
                    settings.threshold(),
                    settings.page_size(),
                    slots,
                    end,
                )
            });
            (
                Some(extension),
                end,
                managers.into_iter().flatten().collect(),
                info,
            )
        } else {
            for record in &mut *records {
                if record.data.is_some() {
                    record.data = record.measure().reserve(&mut allocation)?;
                }
            }
            let layout::ClassicLayout {
                extension: span,
                end,
                manager,
            } = layout::ClassicLayout::finish(
                allocation,
                extension.as_ref().map(|bytes| bytes.len() as u64),
                settings.is_some_and(|settings| settings.persist()),
            )?;
            let info =
                settings
                    .filter(|settings| settings.persist())
                    .map(|settings| match &manager {
                        Some(manager) => file_space_info::persistent_single_manager(
                            super::OFFSET_WIDTH,
                            settings.strategy(),
                            settings.threshold(),
                            settings.page_size(),
                            manager.header_address.get(),
                            end,
                        ),
                        None => file_space_info::persistent_empty(
                            super::OFFSET_WIDTH,
                            settings.strategy(),
                            settings.threshold(),
                            settings.page_size(),
                            end,
                        ),
                    });
            (span, end, manager.into_iter().collect(), info)
        };
        let extension = extension_info
            .as_ref()
            .map(super::file_space_extension_oh)
            .transpose()?
            .or(extension);
        let extension = match (extension_span, extension) {
            (Some(span), Some(bytes)) if span.length == bytes.len() as u64 => Some((span, bytes)),
            (None, None) => None,
            _ => return Err(inconsistent().into()),
        };
        Ok(Self {
            extension,
            end,
            managers,
        })
    }
}

/// Builds hard-link messages with planned header addresses, or zero placeholders.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if a target index is outside `records`. Propagates source link
/// errors.
fn links_at(
    source: &impl ObjectSource,
    index: usize,
    records: Option<&[Record]>,
) -> Result<Vec<LinkMessage>, Error> {
    source
        .links(index)?
        .map(|(name, index)| {
            Ok(super::make_link(
                &name,
                match records {
                    Some(records) => {
                        records
                            .get(index)
                            .ok_or_else(inconsistent)?
                            .metadata
                            .header
                            .address
                    }
                    None if index < source.len() => StoredAddress::ZERO,
                    None => return Err(inconsistent().into()),
                },
            ))
        })
        .collect()
}

/// Writes zero padding in bounded blocks up to the absolute output position `end`.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if output is already past `end`. Propagates sink errors.
fn pad(sink: &mut impl ByteSink<Error = std::io::Error>, end: u64) -> Result<(), Error> {
    if end < sink.position() {
        return Err(inconsistent().into());
    }
    while sink.position() < end {
        sink.put_zeros((end - sink.position()).min(65_536).to_usize()?)?;
    }
    Ok(())
}

/// Checks serialized bytes against their reserved length.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if the lengths differ.
fn check_length(bytes: &[u8], span: Allocation) -> Result<(), FormatError> {
    if bytes.len() as u64 != span.length {
        return Err(inconsistent());
    }
    Ok(())
}

fn emission_error(error: EmissionError<std::io::Error>) -> Error {
    match error {
        EmissionError::Format(error) => Error::Format(error),
        EmissionError::Sink(error) => Error::Io(error),
    }
}

fn inconsistent() -> FormatError {
    FormatError::Internal("source-backed metadata differs from its planned allocation".into())
}

#[cfg(test)]
mod tests {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    use core::mem;
    use core::num::NonZeroUsize;
    use std::borrow::Cow;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use rstest::rstest;
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    use test_util::allocation;

    use super::Allocation;
    use super::DenseLinks;
    use super::Object;
    use super::ObjectSource;
    use super::Plan;
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    use super::Record;
    use crate::AttrValue;
    use crate::DeflateLevel;
    use crate::File;
    use crate::FileSpaceSettings;
    use crate::FileSpaceStrategy;
    use crate::GroupBuilder;
    use crate::LibVer;
    use crate::MaxExtent;
    use crate::address::StoredAddress;
    use crate::address::StoredAddressExt;
    use crate::chunked_write::ChunkMeta;
    use crate::chunked_write::ChunkProvider;
    use crate::data_layout::ChunkIndexLayout;
    use crate::data_layout::ChunkedLayoutFlags;
    use crate::data_layout::DataLayout;
    use crate::error::Error;
    use crate::error::FormatError;
    use crate::file_writer;
    use crate::object_path::ObjectPathBuf;
    use crate::type_builders::DatasetBuilder;

    #[rstest]
    #[case::zero(Some(0))]
    #[case::above_four_gibibytes(Some((1_u64 << 32) + 512))]
    #[case::maximum(None)]
    fn dense_link_geometry_matches_the_full_plan(#[case] address: Option<u64>) {
        let links: Vec<_> = (0..9)
            .map(|index| file_writer::make_link(&format!("d{index:04}"), StoredAddress::new(100)))
            .collect();
        let plan = file_writer::dense_links_plan(&links).unwrap().unwrap();
        let geometry = DenseLinks::new(&plan.link_info(StoredAddress::new(0)).unwrap()).unwrap();
        let span = Allocation {
            address: StoredAddress::new(address.unwrap_or(u64::MAX - plan.serialized_len())),
            length: plan.serialized_len(),
        };
        assert_eq!(
            geometry.link_info(span).unwrap(),
            plan.link_info(span.address).unwrap()
        );
    }

    #[rstest]
    #[case::undefined(true)]
    #[case::overflow(false)]
    fn dense_link_geometry_rejects_a_placement_outside_the_address_width(#[case] undefined: bool) {
        let links: Vec<_> = (0..9)
            .map(|index| file_writer::make_link(&format!("d{index:04}"), StoredAddress::new(100)))
            .collect();
        let plan = file_writer::dense_links_plan(&links).unwrap().unwrap();
        let geometry = DenseLinks::new(&plan.link_info(StoredAddress::new(0)).unwrap()).unwrap();
        let length = plan.serialized_len();
        let address = if undefined {
            u64::MAX
        } else {
            u64::MAX - length + 1
        };
        let error = geometry
            .link_info(Allocation {
                address: StoredAddress::new(address),
                length,
            })
            .unwrap_err();
        if undefined {
            assert_eq!(
                error,
                FormatError::AddressTooLarge {
                    address,
                    offset_size: super::super::OFFSET_WIDTH.get()
                }
            );
        } else {
            assert_eq!(
                error,
                FormatError::OffsetOverflow {
                    offset: address,
                    length
                }
            );
        }
    }

    #[rstest]
    #[case::classic(None)]
    #[case::persistent(Some(FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true)))]
    #[case::page(Some(FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true)))]
    fn empty_index_emission_uses_retained_geometry(
        #[case] settings: Option<FileSpaceSettings>,
        #[values(8, 9, 32, 128)] count: usize,
        #[values(false, true)] filtered: bool,
        #[values(false, true)] unallocated: bool,
    ) {
        let loads = Arc::new((0..=count).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
        let plan = Plan::new(
            EmptyTree {
                count,
                filtered,
                unallocated,
                loads: Arc::clone(&loads),
            },
            settings,
            None,
            LibVer::V110,
        )
        .unwrap();
        let addresses: Vec<_> = plan
            .records
            .iter()
            .take(count)
            .map(|record| {
                assert_eq!(record.empty_index.is_none(), unallocated);
                if unallocated {
                    assert_eq!(record.data.unwrap().length, 0);
                    None
                } else {
                    let allocation = record.data.unwrap();
                    assert!(allocation.length > 0);
                    Some(allocation.address)
                }
            })
            .collect();
        assert_eq!(
            loads
                .iter()
                .take(count)
                .map(|value| value.load(Ordering::Relaxed))
                .collect::<Vec<_>>(),
            vec![1; count]
        );
        let file = File::from_bytes(plan.finish().unwrap()).unwrap();
        assert_eq!(loads[count].load(Ordering::Relaxed), 2);
        assert_eq!(
            loads
                .iter()
                .take(count)
                .map(|value| value.load(Ordering::Relaxed))
                .collect::<Vec<_>>(),
            vec![2; count]
        );
        assert_eq!(file.file_space_settings().unwrap(), settings);
        let mut paths = file.root().datasets().unwrap();
        paths.sort_unstable();
        assert_eq!(
            paths,
            (0..count)
                .map(|index| format!("d{index:04}"))
                .collect::<Vec<_>>()
        );
        for (index, address) in addresses.into_iter().enumerate() {
            let dataset = file.dataset(&format!("d{index:04}")).unwrap();
            assert_eq!(dataset.shape().unwrap(), [0]);
            assert_eq!(dataset.chunk_shape().unwrap(), Some(vec![2]));
            assert_eq!(dataset.read_i32().unwrap(), Vec::<i32>::new());
            assert_eq!(dataset.chunks().unwrap(), []);
            assert_eq!(
                dataset.attrs().unwrap()["index"],
                AttrValue::I32(i32::try_from(index).unwrap())
            );
            let layout = dataset.data_layout().unwrap();
            let DataLayout::Chunked {
                index:
                    ChunkIndexLayout::ExtensibleArray {
                        address: actual, ..
                    },
                ..
            } = layout
            else {
                panic!("expected an extensible array, got {layout:?}");
            };
            assert_eq!(actual, address);
        }
    }

    struct EmptyTree {
        count: usize,
        filtered: bool,
        unallocated: bool,
        loads: Arc<Vec<AtomicUsize>>,
    }

    impl ObjectSource for EmptyTree {
        fn len(&self) -> usize {
            self.count + 1
        }

        fn load(&self, index: usize) -> Result<Object, Error> {
            let Self {
                count,
                filtered,
                unallocated,
                loads,
            } = self;
            loads[index].fetch_add(1, Ordering::Relaxed);
            if index == *count {
                return Ok(Object::group(&GroupBuilder::new("root").finish().attrs)?);
            }
            let mut builder = DatasetBuilder::new(&format!("d{index:04}"));
            if *unallocated {
                builder.with_unallocated_storage(crate::make_i32_type(), &[0]);
            } else {
                builder.with_i32_data(&[]).with_shape(&[0]);
            }
            builder
                .with_chunks(&[2])
                .with_maxshape(&[MaxExtent::Unlimited])
                .set_attr("index", AttrValue::I32(i32::try_from(index).unwrap()));
            if *filtered {
                builder.with_deflate(DeflateLevel::Six);
            }
            Ok(Object::dataset(builder)?)
        }

        fn links(
            &self,
            index: usize,
        ) -> Result<impl ExactSizeIterator<Item = (Cow<'_, str>, usize)>, Error> {
            Ok((0..if index == self.count { self.count } else { 0 })
                .map(|index| (Cow::Owned(format!("d{index:04}")), index)))
        }

        fn resolve(&self, path: &ObjectPathBuf) -> Result<usize, Error> {
            Err(FormatError::UnknownCommittedDatatype(path.to_string()).into())
        }
    }

    #[rstest]
    #[case::first(0, 0)]
    #[case::second(1, 1)]
    #[case::elements(2, 2)]
    #[case::past_end(3, 3)]
    fn collection_lookup_preserves_attribute_and_element_order(
        #[case] index: usize,
        #[case] selected: usize,
    ) {
        let mut builder = DatasetBuilder::new("values");
        builder.with_vlen_strings(&["element"]);
        builder.set_attr("first", AttrValue::VarLenString("first".into()));
        builder.set_attr("empty", AttrValue::VarLenStringArray(Vec::new()));
        builder.set_attr("second", AttrValue::VarLenString("second".into()));
        let object = Object::dataset(builder).unwrap();
        let expected = match selected {
            0 | 1 => object
                .heaps
                .get(selected * 2)
                .and_then(|patch| patch.collections.first()),
            2 => match &object.kind {
                super::Kind::Dataset(dataset) => dataset
                    .storage
                    .vl_string_staging()
                    .and_then(|staging| staging.collections.first()),
                _ => None,
            },
            _ => None,
        };
        assert_eq!(object.collection(index), expected);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[rstest]
    #[case::groups(16, "source_groups", "source_groups_control")]
    #[case::many_groups(64, "source_many_groups", "source_many_groups_control")]
    fn planning_retains_compact_allocation_records(
        #[case] count: usize,
        #[case] region: &'static str,
        #[case] control_region: &'static str,
    ) {
        let (control, positive) =
            allocation::measure(control_region, || std::hint::black_box(vec![0_u8; 1024]));
        assert!(positive.blocks > 0, "{positive}");
        assert!(positive.peak_bytes >= 1024, "{positive}");
        drop(control);
        let (plan, measured) = allocation::measure(region, || {
            Plan::new(AttributeSource { count }, None, None, LibVer::V110).unwrap()
        });
        let retained = (plan.records.capacity() * mem::size_of::<Record>()) as u64;
        assert_eq!(measured.live_bytes, retained, "{measured}");
        assert!(measured.peak_bytes >= 9 * 16_384, "{measured}");
        assert!(measured.peak_bytes <= retained + 1_048_576, "{measured}");
        let file = File::from_bytes(plan.finish().unwrap()).unwrap();
        let mut groups = file.root().groups().unwrap();
        groups.sort_unstable();
        assert_eq!(
            groups,
            (0..count)
                .map(|index| format!("g{index:04}"))
                .collect::<Vec<_>>()
        );
        for index in 0..count {
            let group = file.group(&format!("g{index:04}")).unwrap();
            assert_eq!(group.datasets().unwrap(), Vec::<String>::new());
            assert_eq!(group.groups().unwrap(), Vec::<String>::new());
            let attributes = group.attrs().unwrap();
            assert_eq!(attributes.len(), 9);
            for attribute in 0..9 {
                assert_eq!(
                    attributes[&format!("a{attribute}")],
                    AttrValue::I32Array(vec![i32::try_from(index).unwrap(); 4096])
                );
            }
        }
    }

    #[rstest]
    #[case::classic(None)]
    #[case::direct(Some(FileSpaceSettings::new(FileSpaceStrategy::None)))]
    #[case::persistent(Some(FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true)))]
    #[case::page(Some(FileSpaceSettings::new(FileSpaceStrategy::Page)))]
    #[case::persistent_page(Some(FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true)))]
    fn verbatim_payloads_are_read_once_after_metadata_planning(
        #[case] settings: Option<FileSpaceSettings>,
    ) {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let plan = Plan::new(
            ChunkSource {
                reads: Arc::clone(&reads),
            },
            settings,
            None,
            LibVer::V110,
        )
        .unwrap();
        assert_eq!(*reads.lock().unwrap(), Vec::<usize>::new());
        let file = File::from_bytes(plan.finish().unwrap()).unwrap();
        assert_eq!(*reads.lock().unwrap(), [0, 1]);
        assert_eq!(file.file_space_settings().unwrap(), settings);
        assert_eq!(file.root().datasets().unwrap(), ["values"]);
        let dataset = file.dataset("values").unwrap();
        assert_eq!(dataset.shape().unwrap(), [4]);
        assert_eq!(dataset.chunk_shape().unwrap(), Some(vec![2]));
        assert_eq!(dataset.read_i32().unwrap(), [1, 2, 3, 4]);
    }

    struct ChunkSource {
        reads: Arc<Mutex<Vec<usize>>>,
    }

    impl ObjectSource for ChunkSource {
        fn len(&self) -> usize {
            2
        }

        fn load(&self, index: usize) -> Result<Object, Error> {
            if index == 1 {
                return Ok(Object::group(&GroupBuilder::new("root").finish().attrs)?);
            }
            let mut builder = DatasetBuilder::new("values");
            builder.with_raw_chunks_lazy(
                crate::make_i32_type(),
                &[4],
                None,
                &[2],
                NonZeroUsize::new(4).unwrap(),
                ChunkedLayoutFlags::NONE,
                None,
                vec![
                    ChunkMeta {
                        compressed_size: 8,
                        filter_mask: 0
                    };
                    2
                ],
                Box::new(CountingChunks {
                    reads: Arc::clone(&self.reads),
                }),
            );
            Ok(Object::dataset(builder)?)
        }

        fn links(
            &self,
            index: usize,
        ) -> Result<impl ExactSizeIterator<Item = (Cow<'_, str>, usize)>, Error> {
            Ok((0..usize::from(index == 1)).map(|_| (Cow::Borrowed("values"), 0)))
        }

        fn resolve(&self, path: &ObjectPathBuf) -> Result<usize, Error> {
            Err(FormatError::UnknownCommittedDatatype(path.to_string()).into())
        }
    }

    struct CountingChunks {
        reads: Arc<Mutex<Vec<usize>>>,
    }

    impl ChunkProvider for CountingChunks {
        fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
            let chunks = [[1_i32, 2], [3, 4]];
            let chunk = chunks
                .get(index)
                .ok_or_else(|| FormatError::Internal("the test source has two chunks".into()))?;
            self.reads.lock().unwrap().push(index);
            for value in chunk {
                out.extend_from_slice(&value.to_le_bytes());
            }
            Ok(())
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    struct AttributeSource {
        count: usize,
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    impl ObjectSource for AttributeSource {
        fn len(&self) -> usize {
            self.count + 1
        }

        fn load(&self, index: usize) -> Result<Object, Error> {
            let mut group = GroupBuilder::new("g");
            if index < self.count {
                for attribute in 0..9 {
                    group.set_attr(
                        &format!("a{attribute}"),
                        AttrValue::I32Array(vec![i32::try_from(index).unwrap(); 4096]),
                    );
                }
            }
            Ok(Object::group(&group.finish().attrs)?)
        }

        fn links(
            &self,
            index: usize,
        ) -> Result<impl ExactSizeIterator<Item = (Cow<'_, str>, usize)>, Error> {
            Ok((0..if index == self.count { self.count } else { 0 })
                .map(|index| (Cow::Owned(format!("g{index:04}")), index)))
        }

        fn resolve(&self, path: &ObjectPathBuf) -> Result<usize, Error> {
            Err(FormatError::UnknownCommittedDatatype(path.to_string()).into())
        }
    }
}
