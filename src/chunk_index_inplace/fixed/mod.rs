//! Finite Fixed Array append planning, storage initialization, and ordered publication.

use core::num::NonZeroU64;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::FixedArrayHeader;
use hdf5_pure_format::__private::FixedArrayHeaderLocator;
use hdf5_pure_format::__private::FixedArrayPlacement;
use hdf5_pure_format::__private::FixedArrayPlan;
use hdf5_pure_format::__private::IndexAddressField;
use hdf5_pure_space::__private::AllocationClass;

use crate::address::AddressContext;
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
use crate::edit::allocation::AllocationSession;
use crate::edit::allocation::ConfirmedPublication;
use crate::edit::allocation::ProvisionalAllocation;
use crate::error::Error;
use crate::error::FormatError;
use crate::fill_value::FillPattern;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::ChunkContext;
use crate::filters::FilterScratch;
use crate::object_graph::ObjectId;
use crate::prepared_chunk::ChunkEncoding;
use crate::prepared_chunk::PreparedChunk;
use crate::source::PhysicalSource;
use crate::width::FormatWidths;

use super::ConfirmedChunk;
use super::FiniteGeometry;
use super::ObjectHeaderField;
use super::Store;
use super::WalkedMessage;
use super::metadata;

use self::entries::EntryPatch;

mod entries;

/// A located finite index with retained source encoding, field ranges, and session identity.
///
/// The writer checks current metadata before preparing or publishing an append. Complete source
/// ownership is required when locating the index.
pub(crate) struct FixedIndex {
    session: AllocationSession,
    pub(super) geometry: FiniteGeometry,
    reference: ObjectHeaderField,
    field: IndexAddressField,
    address: Option<StoredAddress>,
    header: Option<FixedArrayHeader>,
    plan: FixedArrayPlan,
}

impl FixedIndex {
    /// Checks the source header against the layout, maximum extent, and complete ownership proof.
    ///
    /// Retains the source entry size and page bits even when its data block is unallocated.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for inconsistent geometry or incomplete/unreachable
    /// ownership. Propagates header, range, encoding, and storage-analysis errors.
    pub(super) fn new(
        file: &impl Store,
        owner: ObjectId,
        geometry: FiniteGeometry,
        message: &WalkedMessage,
        field: IndexAddressField,
        address: Option<StoredAddress>,
        page_bits: u8,
        filtered: bool,
    ) -> Result<Self, Error> {
        let reference = ObjectHeaderField::new(message, field.range())?;
        metadata::require_range(file, reference.chunk)?;
        let entries = geometry.maximum.div_ceil(geometry.chunk_elems);
        let header = address
            .map(|address| read_header(file, address))
            .transpose()?;
        let plan = match header {
            Some(header) => {
                if header.entries() != entries
                    || header.page_bits() != page_bits
                    || header.is_filtered() != filtered
                {
                    return Err(Error::AppendUnsupported(
                        "fixed-array header differs from the dataset's layout or maximum extent",
                    ));
                }
                if let Some(block) = header.data_block() {
                    metadata::require_range(file, block.range())?;
                }
                header.initialization_plan()?
            }
            None => FixedArrayPlan::new(
                entries,
                geometry.chunk_bytes.to_u64(),
                geometry.widths,
                filtered,
            )?
            .with_page_bits(page_bits)?,
        };
        file.require_exclusive_chunk_storage(owner)?;
        Ok(Self {
            session: file.allocation_publication().session(),
            geometry,
            reference,
            field,
            address,
            header,
            plan,
        })
    }

    /// Prepares affected chunk payloads, entry patches, and the new extent before file allocation.
    ///
    /// Reads the affected entry windows and any partial tail. Fills absent chunks from the fill
    /// pattern and checks stored sizes at the retained entry width. The writer retains an
    /// additional filtered version for publication after the extent when completing a lossy raw
    /// edge.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for stale metadata, excessive growth, or forbidden
    /// lossy re-encoding, and [`FormatError::InvalidChunkGeometry`] for inconsistent input lengths.
    /// Propagates source, field-width, fill, filter, range, and buffer-reservation errors.
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
    ) -> Result<FixedAppend, Error> {
        self.require_current(file)?;
        let geometry = &self.geometry;
        let new_dimension =
            geometry
                .current_dim
                .checked_add(additional.get())
                .ok_or(Error::AppendUnsupported(
                    "append would overflow the dataset dimension",
                ))?;
        if new_dimension > geometry.maximum {
            return Err(Error::AppendUnsupported(
                "append exceeds the dataset's maximum first dimension",
            ));
        }
        let mut dimension = [0; 8];
        hdf5_pure_format::__private::write_length_at(
            &mut dimension,
            0,
            new_dimension,
            geometry.widths.lengths,
        )?;
        let expected = additional
            .get()
            .checked_mul(geometry.element_size.get().to_u64())
            .ok_or(FormatError::OffsetOverflow {
                offset: additional.get(),
                length: geometry.element_size.get().to_u64(),
            })?;
        if expected != raw.len().to_u64() {
            return Err(FormatError::InvalidChunkGeometry(
                "append byte count differs from its element count",
            )
            .into());
        }
        let first_slot = geometry.current_dim / geometry.chunk_elems;
        let end_slot = new_dimension.div_ceil(geometry.chunk_elems);
        let (patches, tail) = match self.header.and_then(|header| header.data_block()) {
            Some(block) => {
                let (patches, tail) =
                    EntryPatch::read(file, block, geometry, first_slot, end_slot)?;
                (Some(patches), tail)
            }
            None => (None, None),
        };
        let partial = geometry.current_dim % geometry.chunk_elems;
        if partial != 0
            && !geometry.flags.partial_edge_chunks_stored_raw()
            && pipeline.is_some_and(|pipeline| !edit::pipeline_lossless(pipeline))
            && lossy == LossyTailReencoding::Reject
        {
            return Err(Error::AppendUnsupported(LOSSY_TAIL_REFUSAL));
        }
        let context = ChunkContext::from_datatype(spatial, datatype)?;
        let mut scratch = FilterScratch::new();
        let prefix = match tail.filter(|_| partial != 0) {
            Some(record) => chunk_info(record, first_slot, geometry)?.read_live_elements(
                &PhysicalSource::new(&file.opened_source()).stored(),
                &mut scratch,
                geometry.flags,
                pipeline,
                context,
                geometry.current_dim,
            )?,
            None => Vec::new(),
        };
        let count = (end_slot - first_slot).to_usize()?;
        let mut chunks = Vec::new();
        chunks
            .try_reserve_exact(count)
            .map_err(|source| FormatError::AllocationFailed {
                values: count,
                source,
            })?;
        let mut first = Some(prefix);
        let mut position = 0;
        let mut completed = None;
        for slot in first_slot..end_slot {
            let mut logical = first.take().unwrap_or_default();
            let live = if slot == first_slot { partial } else { 0 };
            let prefix_bytes = live
                .checked_mul(geometry.element_size.get().to_u64())
                .ok_or(FormatError::OffsetOverflow {
                    offset: live,
                    length: geometry.element_size.get().to_u64(),
                })?
                .to_usize()?;
            logical.resize(geometry.chunk_bytes, 0);
            if slot == first_slot && live != 0 && tail.is_none() {
                fill.apply(logical.get_mut(..prefix_bytes).ok_or(
                    FormatError::InvalidChunkGeometry("live prefix exceeds a fixed-array chunk"),
                )?)?;
            }
            let take = (raw.len() - position).min(geometry.chunk_bytes - prefix_bytes);
            logical
                .get_mut(prefix_bytes..prefix_bytes + take)
                .ok_or(FormatError::InvalidChunkGeometry(
                    "appended values exceed a fixed-array chunk",
                ))?
                .copy_from_slice(&raw[position..position + take]);
            position += take;
            if prefix_bytes + take < geometry.chunk_bytes {
                fill.apply(&mut logical[prefix_bytes + take..])?;
            }
            let offset =
                slot.checked_mul(geometry.chunk_elems)
                    .ok_or(FormatError::OffsetOverflow {
                        offset: slot,
                        length: geometry.chunk_elems,
                    })?;
            if slot == first_slot
                && partial != 0
                && geometry.flags.partial_edge_chunks_stored_raw()
                && new_dimension / geometry.chunk_elems > first_slot
                && pipeline.is_some_and(|pipeline| !edit::pipeline_lossless(pipeline))
            {
                completed = Some(PreparedChunk::encode(
                    &mut scratch,
                    pipeline,
                    context,
                    ChunkEncoding::Filtered,
                    logical.clone(),
                )?);
            }
            let current = (offset < geometry.current_dim).then(|| {
                ChunkDecoding::at_extent(
                    geometry.flags,
                    spatial,
                    &[geometry.current_dim],
                    &[offset],
                )
            });
            let encoding = ChunkEncoding::decodable_by(
                current,
                ChunkDecoding::at_extent(geometry.flags, spatial, &[new_dimension], &[offset]),
            );
            chunks.push(PreparedChunk::encode(
                &mut scratch,
                pipeline,
                context,
                encoding,
                logical,
            )?);
        }
        for chunk in chunks.iter().chain(completed.iter()) {
            self.plan.encode_record(
                &mut [0; ENTRY_SCRATCH_BYTES],
                Some(ChunkRecord {
                    address: StoredAddress::new(0),
                    stored_size: chunk.stored_size(),
                    filter_mask: chunk.filter_mask(),
                }),
            )?;
        }
        Ok(FixedAppend {
            session: self.session.clone(),
            widths: geometry.widths,
            base: file.addresses().effective_base(),
            owner: self.reference.field,
            dimension_owner: geometry.dimension.field,
            current_dimension: geometry.current_dim,
            header: self.header,
            address: self.address,
            first_slot,
            end_slot,
            dimension,
            new_dimension,
            chunks,
            completed,
            patches,
        })
    }

    /// Publishes confirmed payloads and index entries before the new extent.
    ///
    /// Rejects stale or foreign plans before allocation. Initializes missing index storage through
    /// the store's placement policy and confirms every allocation before linking references.
    /// A `max_phase` below 2 stops after allocation confirmation. Below 4, stops after index
    /// publication. At phase 4, also publishes the extent and any lossy completion replacement.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for changed session, widths, base, fields, or source
    /// metadata. Propagates allocation, encoding, publication, and synchronization errors.
    pub(crate) fn apply(
        &mut self,
        file: &mut impl Store,
        plan: FixedAppend,
        max_phase: u8,
    ) -> Result<(), Error> {
        let FixedAppend {
            session,
            widths,
            base,
            owner,
            dimension_owner,
            current_dimension,
            header,
            address,
            first_slot,
            end_slot,
            dimension,
            new_dimension,
            chunks,
            completed,
            patches,
        } = plan;
        if !session.same(&file.allocation_publication().session()) || !session.same(&self.session) {
            return Err(Error::AppendUnsupported(
                "append plan refers to a different editing session",
            ));
        }
        if widths != file.widths() || widths != self.geometry.widths {
            return Err(Error::AppendUnsupported(
                "append field widths changed after planning",
            ));
        }
        if base != file.addresses().effective_base()
            || owner != self.reference.field
            || dimension_owner != self.geometry.dimension.field
        {
            return Err(Error::AppendUnsupported(
                "append plan refers to a different chunk index",
            ));
        }
        if current_dimension != self.geometry.current_dim
            || header != self.header
            || address != self.address
        {
            return Err(Error::AppendUnsupported(
                "chunk reference changed after append planning",
            ));
        }
        self.require_current(file)?;
        if let Some(patches) = &patches {
            for patch in patches {
                patch.require_current(file)?;
            }
        }
        let allocations = chunks
            .iter()
            .map(|chunk| file.allocate(chunk.bytes(), AllocationClass::Raw))
            .collect::<Result<Vec<_>, _>>()?;
        let completion = completed
            .as_ref()
            .map(|chunk| file.allocate(chunk.bytes(), AllocationClass::Raw))
            .transpose()?;
        let initialization = self.reserve_initialization(file)?;
        let publication = file.publish_allocations()?;
        let records = allocations
            .into_iter()
            .zip(&chunks)
            .map(|(allocation, chunk)| confirmed_record(file, &publication, allocation, chunk))
            .collect::<Result<Vec<_>, _>>()?;
        let completion = completion
            .zip(completed.as_ref())
            .map(|(allocation, chunk)| confirmed_record(file, &publication, allocation, chunk))
            .transpose()?;
        if max_phase < 2 {
            return Ok(());
        }
        if let Some(initialization) = initialization {
            self.publish_initialization(file, &publication, initialization)?;
        }
        let patches = match patches {
            Some(patches) => patches,
            None => {
                let block = self
                    .header
                    .and_then(|header| header.data_block())
                    .ok_or_else(|| {
                        FormatError::Internal("initialized fixed-array data block is absent".into())
                    })?;
                EntryPatch::read(file, block, &self.geometry, first_slot, end_slot)?.0
            }
        };
        publish_patches(file, patches, first_slot, &records)?;
        if max_phase < 4 {
            return Ok(());
        }
        self.geometry
            .dimension
            .publish(file, &dimension[..usize::from(widths.lengths.get())])?;
        file.sync()?;
        self.geometry.current_dim = new_dimension;
        if let Some(record) = completion {
            let block = self
                .header
                .and_then(|header| header.data_block())
                .ok_or_else(|| {
                    FormatError::Internal("completed fixed-array data block is absent".into())
                })?;
            let patches =
                EntryPatch::read(file, block, &self.geometry, first_slot, first_slot + 1)?.0;
            publish_patches(file, patches, first_slot, &[record])?;
        }
        Ok(())
    }

    /// Checks the session, recorded widths, header, layout reference, and current extent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if a retained fact changed. Propagates field, header,
    /// source, and range errors.
    fn require_current(&self, file: &impl Store) -> Result<(), Error> {
        if !self.session.same(&file.allocation_publication().session()) {
            return Err(Error::AppendUnsupported(
                "append file session differs from the located dataset",
            ));
        }
        if file.widths() != self.geometry.widths {
            return Err(Error::AppendUnsupported(
                "append file widths differ from the located dataset",
            ));
        }
        metadata::require_range(file, self.reference.chunk)?;
        metadata::require_range(file, self.geometry.dimension.chunk)?;
        let expected = self.field.encode(self.address)?;
        let mut actual = [0; 8];
        let length = expected.as_bytes().len();
        file.read_at(self.reference.field.start().get(), &mut actual[..length])?;
        if &actual[..length] != expected.as_bytes() {
            return Err(Error::AppendUnsupported(
                "chunk reference changed after append planning",
            ));
        }
        if self
            .address
            .map(|address| read_header(file, address))
            .transpose()?
            != self.header
        {
            return Err(Error::AppendUnsupported(
                "fixed-array header changed after append planning",
            ));
        }
        let mut expected = [0; 8];
        hdf5_pure_format::__private::write_length_at(
            &mut expected,
            0,
            self.geometry.current_dim,
            self.geometry.widths.lengths,
        )?;
        let length = usize::from(self.geometry.widths.lengths.get());
        file.read_at(
            self.geometry.dimension.field.start().get(),
            &mut actual[..length],
        )?;
        if actual[..length] != expected[..length] {
            return Err(Error::AppendUnsupported(
                "dataspace extent changed after append planning",
            ));
        }
        Ok(())
    }

    /// Reserves missing index storage through the store's allocation policy.
    ///
    /// Returns `None` when the data block is already allocated.
    ///
    /// # Errors
    ///
    /// Propagates initialization planning, reservation, emission, and provisioning errors.
    fn reserve_initialization(
        &self,
        file: &mut impl Store,
    ) -> Result<Option<FixedInitialization>, Error> {
        if self
            .header
            .is_some_and(|header| header.data_block().is_some())
        {
            return Ok(None);
        }
        file.allocate_fixed_initialization(&self.plan, self.header)
            .map(Some)
    }

    /// Confirms complete storage before publishing a new header or data-block reference.
    ///
    /// Rereads the published header after the reference barrier. A failed reread marks allocation
    /// publication uncertain.
    ///
    /// # Errors
    ///
    /// Propagates ownership confirmation, field encoding, publication, and header-read errors.
    fn publish_initialization(
        &mut self,
        file: &mut impl Store,
        publication: &ConfirmedPublication,
        pending: FixedInitialization,
    ) -> Result<(), Error> {
        let combined = self.header.is_none();
        let placement = pending.confirm(file, publication, combined)?;
        let (header, _) = placement
            .blocks()
            .next()
            .ok_or(FormatError::InvalidFixedArrayGeometry)?;
        if combined {
            self.reference
                .publish(file, self.field.encode(Some(header))?.as_bytes())?;
        } else {
            file.write_at(
                file.addresses().resolve(header)?.get(),
                &placement.serialize_header()?,
            )?;
        }
        file.sync()?;
        let parsed = read_header(file, header);
        if parsed.is_err() {
            file.make_allocation_uncertain();
        }
        self.header = Some(parsed?);
        self.address = Some(header);
        Ok(())
    }
}

/// Prepared chunks and entry patches bound to a session, dataset, and recorded source state.
pub(crate) struct FixedAppend {
    session: AllocationSession,
    widths: FormatWidths,
    base: BaseAddress,
    owner: FileRange,
    dimension_owner: FileRange,
    current_dimension: u64,
    header: Option<FixedArrayHeader>,
    address: Option<StoredAddress>,
    first_slot: u64,
    end_slot: u64,
    dimension: [u8; 8],
    new_dimension: u64,
    chunks: Vec<PreparedChunk>,
    completed: Option<PreparedChunk>,
    patches: Option<Vec<EntryPatch>>,
}

/// Placed index blocks paired with their complete provisional storage allocations.
///
/// Storage may be combined or reserved as separate blocks. Every supplied token is confirmed
/// against the placement before publishing a reference.
pub(crate) struct FixedInitialization {
    placement: FixedArrayPlacement,
    storage: InitializationStorage,
}

impl FixedInitialization {
    /// Emits missing storage into one combined reservation, retaining a source header if present.
    ///
    /// Stores may provide separate block allocation through
    /// [`Store::allocate_fixed_initialization`].
    ///
    /// # Errors
    ///
    /// Propagates plan, placement, emission, and allocation errors. Returns
    /// [`FormatError::Internal`]
    /// if the returned reservation differs from the placed blocks.
    pub(crate) fn allocate(
        file: &mut impl Store,
        plan: &FixedArrayPlan,
        header: Option<FixedArrayHeader>,
    ) -> Result<Self, Error> {
        let combined = header.is_none();
        let length = if combined {
            plan.total_len()
        } else {
            plan.block_lengths().nth(1).ok_or_else(|| {
                FormatError::Internal("fixed-array data-block length is absent".into())
            })?
        };
        let length = NonZeroU64::new(length).ok_or(FormatError::InvalidFixedArrayGeometry)?;
        let addresses = file.addresses();
        let mut placement = None;
        let allocation =
            file.allocate_emitted(length, AllocationClass::Metadata, |address, write| {
                let placed = match header {
                    Some(header) => header.initialization_at(addresses.relative(address)?)?,
                    None => {
                        let header_length = plan
                            .block_lengths()
                            .next()
                            .ok_or(FormatError::InvalidFixedArrayGeometry)?;
                        let header = addresses.relative(address)?;
                        plan.place(header, header.checked_add(header_length)?)?
                    }
                };
                if combined {
                    write(&placed.serialize_header()?)?;
                }
                placed.emit_empty_data_block(&mut *write)?;
                placement = Some(placed);
                Ok(())
            })?;
        let placement = placement.ok_or_else(|| {
            FormatError::Internal("fixed-array placement is absent after emission".into())
        })?;
        if combined {
            Ok(Self {
                placement,
                storage: InitializationStorage::Combined(allocation),
            })
        } else {
            Self::from_blocks(addresses, placement, None, allocation)
        }
    }

    /// Checks separate provisional allocations against complete placed block ranges.
    ///
    /// `header` is absent when retaining an existing source header.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if a reservation differs from its placed block.
    /// Propagates placement-range and address-conversion errors.
    pub(crate) fn from_blocks(
        addresses: AddressContext,
        placement: FixedArrayPlacement,
        header: Option<ProvisionalAllocation>,
        block: ProvisionalAllocation,
    ) -> Result<Self, Error> {
        let [header_range, block_range] = Self::ranges(addresses, &placement)?;
        if block.range() != block_range
            || header
                .as_ref()
                .is_some_and(|header| header.range() != header_range)
        {
            return Err(FormatError::Internal(
                "fixed-array reservations differ from their placement".into(),
            )
            .into());
        }
        Ok(Self {
            placement,
            storage: InitializationStorage::Blocks { header, block },
        })
    }

    fn ranges(
        addresses: AddressContext,
        placement: &FixedArrayPlacement,
    ) -> Result<[FileRange; 2], Error> {
        let mut blocks = placement.blocks();
        let (header, header_length) = blocks
            .next()
            .ok_or(FormatError::InvalidFixedArrayGeometry)?;
        let (block, block_length) = blocks
            .next()
            .ok_or(FormatError::InvalidFixedArrayGeometry)?;
        Ok([
            FileRange::new(addresses.resolve(header)?, header_length)?,
            FileRange::new(addresses.resolve(block)?, block_length)?,
        ])
    }

    fn confirm_range(
        file: &impl Store,
        publication: &ConfirmedPublication,
        allocation: ProvisionalAllocation,
        expected: FileRange,
    ) -> Result<(), Error> {
        let allocation = publication.confirm(allocation)?;
        if file.allocation_publication().require(&allocation)? != expected {
            return Err(FormatError::Internal(
                "fixed-array reservation differs from its placement".into(),
            )
            .into());
        }
        Ok(())
    }

    /// Confirms every provisional block and requires the expected header-allocation form.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] for inconsistent allocation forms or ranges. Propagates
    /// confirmation and placement-range errors.
    fn confirm(
        self,
        file: &impl Store,
        publication: &ConfirmedPublication,
        new_header: bool,
    ) -> Result<FixedArrayPlacement, Error> {
        let Self { placement, storage } = self;
        let [header_range, block_range] = Self::ranges(file.addresses(), &placement)?;
        match storage {
            InitializationStorage::Combined(allocation) => {
                if !new_header || header_range.end() != block_range.start() {
                    return Err(FormatError::Internal(
                        "combined fixed-array storage differs from its blocks".into(),
                    )
                    .into());
                }
                Self::confirm_range(
                    file,
                    publication,
                    allocation,
                    FileRange::new(
                        header_range.start(),
                        header_range
                            .length()
                            .checked_add(block_range.length())
                            .ok_or(FormatError::InvalidFixedArrayGeometry)?,
                    )?,
                )?;
            }
            InitializationStorage::Blocks { header, block } => {
                if new_header != header.is_some() {
                    return Err(FormatError::Internal(
                        "fixed-array header allocation differs from initialization".into(),
                    )
                    .into());
                }
                if let Some(header) = header {
                    Self::confirm_range(file, publication, header, header_range)?;
                }
                Self::confirm_range(file, publication, block, block_range)?;
            }
        }
        Ok(placement)
    }
}

/// Provisional ownership for combined storage or independently placed metadata blocks.
enum InitializationStorage {
    Blocks {
        header: Option<ProvisionalAllocation>,
        block: ProvisionalAllocation,
    },
    Combined(ProvisionalAllocation),
}

/// Reads a header after checking its complete range and checksum.
///
/// # Errors
///
/// Propagates source, range, checksum, and format errors.
fn read_header(file: &impl Store, address: StoredAddress) -> Result<FixedArrayHeader, Error> {
    let locator =
        FixedArrayHeaderLocator::new(file.widths(), file.addresses().effective_base(), address);
    let range = locator.parse_range()?;
    metadata::verify_range(file, range)?;
    let bytes = file.read_metadata_at(range.start().get(), range.length().to_usize()?)?;
    Ok(locator.parse(FileRegion::new(range, &bytes)?)?)
}

fn confirmed_record(
    file: &impl Store,
    publication: &ConfirmedPublication,
    allocation: ProvisionalAllocation,
    chunk: &PreparedChunk,
) -> Result<ChunkRecord, Error> {
    let record = ConfirmedChunk {
        allocation: publication.confirm(allocation)?,
        filter_mask: chunk.filter_mask(),
    }
    .record(file)?;
    Ok(ChunkRecord {
        address: record.addr,
        stored_size: record.stored_size,
        filter_mask: record.filter_mask,
    })
}

fn chunk_info(
    record: ChunkRecord,
    slot: u64,
    geometry: &FiniteGeometry,
) -> Result<ChunkInfo, FormatError> {
    Ok(ChunkInfo {
        address: record.address,
        chunk_size: StoredChunkSize::v4(record.stored_size),
        filter_mask: record.filter_mask,
        offsets: vec![slot.checked_mul(geometry.chunk_elems).ok_or(
            FormatError::OffsetOverflow {
                offset: slot,
                length: geometry.chunk_elems,
            },
        )?],
    })
}

/// Publishes affected entry windows, then publishes initialized-page bits in one batch.
///
/// # Errors
///
/// Propagates patch, checksum, synchronization, and range errors.
fn publish_patches(
    file: &mut impl Store,
    mut patches: Vec<EntryPatch>,
    first: u64,
    records: &[ChunkRecord],
) -> Result<(), Error> {
    for patch in &mut patches {
        let start = (patch.first_slot() - first).to_usize()?;
        let length = patch.entries()?;
        let selected = records
            .get(
                start
                    ..start
                        .checked_add(length)
                        .ok_or(FormatError::InvalidFixedArrayGeometry)?,
            )
            .ok_or_else(|| {
                FormatError::Internal("fixed-array records do not cover their patch".into())
            })?;
        patch.publish(file, selected)?;
    }
    EntryPatch::publish_initializations(file, &patches)
}

const ENTRY_SCRATCH_BYTES: usize = 32;
