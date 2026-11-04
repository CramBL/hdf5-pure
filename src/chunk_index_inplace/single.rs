//! Prepared replacement chunks and ordered publication for finite Single Chunk appends.

use core::num::NonZeroU64;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::SingleChunkField;
use hdf5_pure_space::__private::AllocationClass;

use crate::address::BaseAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunked_read::ChunkDecoding;
use crate::chunked_read::ChunkInfo;
use crate::chunked_read::StoredChunkSize;
use crate::convert::Narrow;
use crate::datatype::Datatype;
use crate::edit;
use crate::edit::LOSSY_TAIL_REFUSAL;
use crate::edit::LossyTailReencoding;
use crate::edit::retirement::ChunkAllocation;
use crate::error::Error;
use crate::error::FormatError;
use crate::fill_value::FillPattern;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::ChunkContext;
use crate::filters::FilterScratch;
use crate::prepared_chunk::ChunkEncoding;
use crate::prepared_chunk::PreparedChunk;
use crate::source::PhysicalSource;
use crate::width::FormatWidths;

use super::ConfirmedChunk;
use super::FiniteGeometry;
use super::ObjectHeaderField;
use super::RetiredChunk;
use super::Store;
use super::WalkedMessage;
use super::metadata;

/// A located Single Chunk reference with checked geometry and source ownership.
///
/// The writer retains the encoded reference to reject stale plans. Allocation ownership is
/// confirmed separately before publishing each replacement.
pub(crate) struct SingleIndex {
    pub(super) geometry: FiniteGeometry,
    field: SingleChunkField,
    reference: ObjectHeaderField,
    bytes: ReferenceBytes,
    record: Option<ChunkRecord>,
}

impl SingleIndex {
    /// Checks local reference ranges and requires complete, reachable source ownership.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if the maximum spans more than one chunk,
    /// or [`Error::AppendUnsupported`] if chunk storage overlaps its object header, ownership is
    /// incomplete, or the header is unreachable. Propagates field, range, and metadata errors.
    pub(super) fn new(
        file: &impl Store,
        header: u64,
        geometry: FiniteGeometry,
        message: &WalkedMessage,
        field: SingleChunkField,
        layout: &[u8],
        record: Option<ChunkRecord>,
    ) -> Result<Self, Error> {
        if geometry.maximum > geometry.chunk_elems {
            return Err(FormatError::InvalidChunkGeometry(
                "a single-chunk maximum spans more than one chunk",
            )
            .into());
        }
        let bytes = ReferenceBytes::new(layout.get(field.range()).ok_or_else(|| {
            FormatError::Internal("a single-chunk reference lies outside its layout".into())
        })?)?;
        let reference = ObjectHeaderField::new(message, field.range())?;
        metadata::require_range(file, reference.chunk)?;
        let index = Self {
            geometry,
            field,
            reference,
            bytes,
            record,
        };
        if let Some(record) = record {
            index.capture(file, record)?;
        }
        file.require_exclusive_chunk_storage(header)?;
        Ok(index)
    }

    /// Prepares a replacement chunk and encoded extent without allocating file storage.
    ///
    /// Reads existing live values or fills an unallocated prefix, then appends `raw` and fills
    /// the remaining chunk. Retains a second filtered version when completing a raw partial
    /// edge under a lossy pipeline requires publication at both extents.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for changed widths or metadata, an overflowing or
    /// excessive extent, or forbidden lossy re-encoding. Returns
    /// [`FormatError::InvalidChunkGeometry`] for inconsistent input length, or
    /// [`FormatError::LengthTooLarge`] if an extent or filtered size exceeds its encoded width.
    /// Propagates source, range, fill, and filter errors.
    pub(crate) fn plan(
        &self,
        file: &impl Store,
        datatype: &Datatype,
        spatial: &[u64],
        pipeline: Option<&FilterPipeline>,
        lossy: LossyTailReencoding,
        raw: &[u8],
        additional: NonZeroU64,
        fill: FillPattern<'_>,
    ) -> Result<SingleAppend, Error> {
        let FiniteGeometry {
            widths,
            dimension: _,
            current_dim,
            maximum,
            flags,
            chunk_elems,
            element_size,
            chunk_bytes,
        } = self.geometry;
        if widths != file.widths() {
            return Err(Error::AppendUnsupported(
                "append file widths differ from the located dataset",
            ));
        }
        let new_dim = current_dim
            .checked_add(additional.get())
            .ok_or(Error::AppendUnsupported(
                "append would overflow the dataset dimension",
            ))?;
        if new_dim > maximum {
            return Err(Error::AppendUnsupported(
                "append exceeds the dataset's maximum first dimension",
            ));
        }
        let mut dimension = [0; 8];
        hdf5_pure_format::__private::write_length_at(&mut dimension, 0, new_dim, widths.lengths)?;
        let old_length = current_dim
            .checked_mul(element_size.get().to_u64())
            .ok_or(FormatError::OffsetOverflow {
                offset: current_dim,
                length: element_size.get().to_u64(),
            })?
            .to_usize()?;
        let new_length = new_dim
            .checked_mul(element_size.get().to_u64())
            .ok_or(FormatError::OffsetOverflow {
                offset: new_dim,
                length: element_size.get().to_u64(),
            })?
            .to_usize()?;
        if raw.len() != new_length - old_length {
            return Err(FormatError::InvalidChunkGeometry(
                "append byte count differs from its element count",
            )
            .into());
        }
        self.require_current(file)?;
        if current_dim != 0
            && !flags.partial_edge_chunks_stored_raw()
            && pipeline.is_some_and(|pipeline| !edit::pipeline_lossless(pipeline))
            && lossy == LossyTailReencoding::Reject
        {
            return Err(Error::AppendUnsupported(LOSSY_TAIL_REFUSAL));
        }
        let predecessor = self
            .record
            .map(|record| self.capture(file, record))
            .transpose()?;
        let context = ChunkContext::from_datatype(spatial, datatype)?;
        let mut scratch = FilterScratch::new();
        let mut logical = match self.record.filter(|_| current_dim > 0) {
            Some(record) => self.chunk(record).read_live_elements(
                &PhysicalSource::new(&file.opened_source()).stored(),
                &mut scratch,
                flags,
                pipeline,
                context,
                current_dim,
            )?,
            None => Vec::new(),
        };
        logical.resize(chunk_bytes, 0);
        if self.record.is_none() && old_length > 0 {
            fill.apply(
                logical
                    .get_mut(..old_length)
                    .ok_or(FormatError::InvalidChunkGeometry(
                        "live prefix exceeds a single chunk",
                    ))?,
            )?;
        }
        logical
            .get_mut(old_length..new_length)
            .ok_or(FormatError::InvalidChunkGeometry(
                "appended values exceed a single chunk",
            ))?
            .copy_from_slice(raw);
        if new_length < chunk_bytes {
            fill.apply(
                logical
                    .get_mut(new_length..)
                    .ok_or(FormatError::InvalidChunkGeometry(
                        "appended extent exceeds a single chunk",
                    ))?,
            )?;
        }
        let completed = if current_dim > 0
            && flags.partial_edge_chunks_stored_raw()
            && new_dim == chunk_elems
            && pipeline.is_some_and(|pipeline| !edit::pipeline_lossless(pipeline))
        {
            Some(PreparedChunk::encode(
                &mut scratch,
                pipeline,
                context,
                ChunkEncoding::Filtered,
                logical.clone(),
            )?)
        } else {
            None
        };
        let current = (current_dim > 0)
            .then(|| ChunkDecoding::at_extent(flags, &[chunk_elems], &[current_dim], &[0]));
        let encoding = ChunkEncoding::decodable_by(
            current,
            ChunkDecoding::at_extent(flags, &[chunk_elems], &[new_dim], &[0]),
        );
        let chunk = PreparedChunk::encode(&mut scratch, pipeline, context, encoding, logical)?;
        for chunk in core::iter::once(&chunk).chain(completed.iter()) {
            self.field.encode(Some(ChunkRecord {
                address: StoredAddress::new(0),
                stored_size: chunk.stored_size(),
                filter_mask: chunk.filter_mask(),
            }))?;
        }
        Ok(SingleAppend {
            widths,
            reference: self.reference.field,
            base: file.addresses().effective_base(),
            current_dim,
            bytes: self.bytes,
            dimension,
            new_dim,
            chunk,
            completed,
            predecessor,
        })
    }

    /// Publishes a prepared replacement before its new extent, then retires confirmed storage.
    ///
    /// Checks plan identity and current metadata before allocation. When `max_phase` is below 2,
    /// stops after allocation confirmation. Below 4, stops after reference publication. At phase 4,
    /// also publishes the extent and any final filtered replacement.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for changed widths, base, index identity, reference,
    /// or extent. Propagates allocation, encoding, publication, range, and synchronization errors.
    pub(crate) fn apply(
        &mut self,
        file: &mut impl Store,
        plan: SingleAppend,
        max_phase: u8,
    ) -> Result<(), Error> {
        let SingleAppend {
            widths,
            reference,
            base,
            current_dim,
            bytes,
            dimension,
            new_dim,
            chunk,
            completed,
            predecessor,
        } = plan;
        if widths != file.widths() || widths != self.geometry.widths {
            return Err(Error::AppendUnsupported(
                "append field widths changed after planning",
            ));
        }
        if reference != self.reference.field || base != file.addresses().effective_base() {
            return Err(Error::AppendUnsupported(
                "append plan refers to a different chunk index",
            ));
        }
        if current_dim != self.geometry.current_dim || bytes != self.bytes {
            return Err(Error::AppendUnsupported(
                "chunk reference changed after append planning",
            ));
        }
        self.require_current(file)?;
        let allocation = file.allocate(chunk.bytes(), AllocationClass::Raw)?;
        let completion = completed
            .as_ref()
            .map(|chunk| file.allocate(chunk.bytes(), AllocationClass::Raw))
            .transpose()?;
        let publication = file.publish_allocations()?;
        let replacement = ConfirmedChunk {
            allocation: publication.confirm(allocation)?,
            filter_mask: chunk.filter_mask(),
        };
        let completion = completion
            .zip(completed.as_ref())
            .map(|(allocation, chunk)| {
                publication
                    .confirm(allocation)
                    .map(|allocation| ConfirmedChunk {
                        allocation,
                        filter_mask: chunk.filter_mask(),
                    })
            })
            .transpose()?;
        if max_phase < 2 {
            return Ok(());
        }
        self.replace(file, &replacement, predecessor)?;
        if max_phase < 4 {
            return Ok(());
        }
        self.geometry
            .dimension
            .publish(file, &dimension[..usize::from(widths.lengths.get())])?;
        file.sync()?;
        self.geometry.current_dim = new_dim;
        if let Some(completion) = completion {
            let record = self
                .record
                .ok_or_else(|| FormatError::Internal("completed single chunk is absent".into()))?;
            let predecessor = self.capture(file, record)?;
            self.replace(file, &completion, Some(predecessor))?;
        }
        Ok(())
    }

    /// Publishes a confirmed chunk reference and retires its predecessor after the barrier.
    ///
    /// # Errors
    ///
    /// Propagates confirmation, encoding, metadata publication, and synchronization errors.
    fn replace(
        &mut self,
        file: &mut impl Store,
        replacement: &ConfirmedChunk,
        predecessor: Option<ChunkAllocation>,
    ) -> Result<(), Error> {
        let record = replacement.record(file)?;
        let record = ChunkRecord {
            address: record.addr,
            stored_size: record.stored_size,
            filter_mask: record.filter_mask,
        };
        let encoded = self.field.encode(Some(record))?;
        let bytes = ReferenceBytes::new(encoded.as_bytes())?;
        self.reference.publish(file, encoded.as_bytes())?;
        file.sync()?;
        self.record = Some(record);
        self.bytes = bytes;
        if let Some(allocation) = predecessor {
            file.retire(RetiredChunk { allocation });
        }
        Ok(())
    }

    /// Checks both containing metadata ranges and their recorded reference and extent bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if either field changed, and propagates source,
    /// range, and encoding errors.
    fn require_current(&self, file: &impl Store) -> Result<(), Error> {
        metadata::require_range(file, self.reference.chunk)?;
        metadata::require_range(file, self.geometry.dimension.chunk)?;
        let mut bytes = self.bytes;
        file.read_at(self.reference.field.start().get(), bytes.as_mut_slice()?)?;
        if bytes != self.bytes {
            return Err(Error::AppendUnsupported(
                "chunk reference changed after append planning",
            ));
        }
        let mut expected = [0; 8];
        hdf5_pure_format::__private::write_length_at(
            &mut expected,
            0,
            self.geometry.current_dim,
            self.geometry.widths.lengths,
        )?;
        let mut actual = [0; 8];
        let length = usize::from(self.geometry.widths.lengths.get());
        file.read_at(
            self.geometry.dimension.field.start().get(),
            &mut actual[..length],
        )?;
        if actual != expected {
            return Err(Error::AppendUnsupported(
                "dataspace extent changed after append planning",
            ));
        }
        Ok(())
    }

    /// Captures an in-bounds predecessor range disjoint from the dataset's metadata chunks.
    ///
    /// Complete ownership analysis is required when locating the index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if the range overlaps either metadata chunk.
    /// Propagates address-conversion and range errors.
    fn capture(&self, file: &impl Store, record: ChunkRecord) -> Result<ChunkAllocation, Error> {
        let allocation = ChunkAllocation::capture(file.addresses(), &self.chunk(record))?;
        metadata::require_range(file, allocation.range())?;
        if allocation.range().overlaps(self.reference.chunk)
            || allocation.range().overlaps(self.geometry.dimension.chunk)
        {
            return Err(Error::AppendUnsupported(
                "chunk storage overlaps its dataset's object header",
            ));
        }
        Ok(allocation)
    }

    fn chunk(&self, record: ChunkRecord) -> ChunkInfo {
        ChunkInfo {
            chunk_size: StoredChunkSize::v4(record.stored_size),
            filter_mask: record.filter_mask,
            offsets: vec![0],
            address: record.address,
        }
    }
}

/// Prepared payloads, field bytes, and predecessor facts bound to one located dataset.
pub(crate) struct SingleAppend {
    widths: FormatWidths,
    reference: FileRange,
    base: BaseAddress,
    current_dim: u64,
    bytes: ReferenceBytes,
    dimension: [u8; 8],
    new_dim: u64,
    chunk: PreparedChunk,
    /// The final filtered chunk when extent publication requires an intermediate raw chunk.
    completed: Option<PreparedChunk>,
    predecessor: Option<ChunkAllocation>,
}

/// A bounded copy of a stored reference for stale-field checks.
#[derive(Clone, Copy, Eq, PartialEq)]
struct ReferenceBytes {
    bytes: [u8; REFERENCE_READ_BYTES],
    length: usize,
}

impl ReferenceBytes {
    /// Copies a stored reference for later stale-field comparisons.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the reference exceeds the fixed buffer.
    fn new(input: &[u8]) -> Result<Self, FormatError> {
        let mut output = Self {
            bytes: [0; REFERENCE_READ_BYTES],
            length: input.len(),
        };
        output.as_mut_slice()?.copy_from_slice(input);
        Ok(output)
    }

    /// Borrows the recorded reference prefix, excluding unused capacity.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the recorded length exceeds the buffer.
    fn as_mut_slice(&mut self) -> Result<&mut [u8], FormatError> {
        self.bytes.get_mut(..self.length).ok_or_else(|| {
            FormatError::Internal("a layout reference exceeds its read buffer".into())
        })
    }
}

const REFERENCE_READ_BYTES: usize = 32;
