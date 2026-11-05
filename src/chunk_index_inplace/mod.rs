//! Immediate mutation of Extensible Array, Fixed Array, and Single Chunk indexes.
//!
//! The edit engine supplies allocation, I/O, and publication through [`Store`]. Extensible
//! Arrays are extended with data and secondary blocks. The writer replaces a Single Chunk
//! separately from its predecessor and patches affected Fixed Array entries. The writer confirms
//! allocation ownership before publishing a
//! reference and publishes the new extent after the reference's ordering barrier.
//!
//! [`Dataset::append`](crate::Dataset::append) supports partial tails and re-encodable filters.
//! SWMR appends require an unfiltered Extensible Array and chunk-aligned extents.

use core::num::NonZeroUsize;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::NonZeroByteLength;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::BlockOffsetWidth;
use hdf5_pure_format::__private::ChunkIndexFields;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::ChunkedGeometry;
use hdf5_pure_format::__private::DataBlockGeometry;
use hdf5_pure_format::__private::EA_CLIENT_ID_UNFILTERED;
use hdf5_pure_format::__private::ExtensibleArray;
use hdf5_pure_format::__private::ExtensibleArrayBitGeometry;
use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;
use hdf5_pure_format::__private::ExtensibleArrayGeometry;
use hdf5_pure_format::__private::ExtensibleArrayHeader;
use hdf5_pure_format::__private::ExtensibleArrayInitialization;
use hdf5_pure_format::__private::ExtensibleArrayStats;
use hdf5_pure_format::__private::FixedArrayHeader;
use hdf5_pure_format::__private::FixedArrayPlan;
use hdf5_pure_format::__private::IndexSlots;
use hdf5_pure_format::__private::MessageRecord;
use hdf5_pure_format::__private::MessageRecordLayout;
use hdf5_pure_format::__private::OBJECT_HEADER_PREFIX_MAX_LEN;
use hdf5_pure_format::__private::ObjectHeaderContinuation;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use hdf5_pure_format::__private::SuperBlockGeometry;
use hdf5_pure_space::__private::{AllocationClass, Extent, FreeClass};

#[cfg(test)]
use crate::DeflateLevel;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::address::{AddressContext, BaseAddress};
use crate::checksum;
use crate::chunked_read;
use crate::chunked_read::ChunkDecoding;
use crate::chunked_read::ChunkInfo;
use crate::chunked_read::StoredChunkSize;
use crate::chunked_write;
use crate::convert::Narrow;
use crate::data_layout::ChunkIndexLayout;
use crate::data_layout::ChunkedLayoutFlags;
use crate::data_layout::DataLayout;
use crate::dataspace::{Dataspace, MaxExtent};
use crate::datatype::Datatype;
use crate::edit::LossyTailReencoding;
use crate::edit::allocation::{
    AllocationPublication, ConfirmedAllocation, ConfirmedPublication, ProvisionalAllocation,
};
use crate::edit::retirement::ChunkAllocation;
use crate::edit::{self, LOSSY_TAIL_REFUSAL};
use crate::error::{Error, FormatError};
use crate::fill_value::FillPattern;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::ChunkContext;
use crate::filters::FilterScratch;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_graph::ObjectId;
use crate::prepared_chunk::ChunkEncoding;
use crate::prepared_chunk::PreparedChunk;
use crate::reader::OpenedSource;
#[cfg(test)]
use crate::source;
#[cfg(test)]
use crate::source::PhysicalBytes;
use crate::source::{PhysicalSource, Source};
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

mod fixed;
mod metadata;
mod single;

pub(crate) use fixed::FixedAppend;
pub(crate) use fixed::FixedIndex;
pub(crate) use fixed::FixedInitialization;
pub(crate) use single::SingleAppend;
pub(crate) use single::SingleIndex;

/// Counts Extensible-Array data-block and super-block allocations on the current thread.
///
/// Crash-prefix tests require allocations within the recorded window to exercise the
/// allocation-specific ordering barriers.
#[cfg(test)]
pub(crate) mod alloc_probe;

/// One stored Extensible-Array element: the chunk address, plus (for filtered
/// arrays) the stored/compressed chunk size and this chunk's filter mask. The
/// size and mask are ignored for unfiltered arrays (client id 0).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ElemRecord {
    pub addr: StoredAddress,
    pub stored_size: u64,
    pub filter_mask: u32,
}

impl ElemRecord {
    /// A bare-address record (the size/mask are unused for unfiltered arrays).
    pub(crate) fn addr_only(addr: StoredAddress) -> Self {
        Self {
            addr,
            stored_size: 0,
            filter_mask: 0,
        }
    }
}

/// Allocation publication and I/O for the shared immediate-append writers.
///
/// The edit engine supplies this interface for a whole-file mirror or a bounded handle image.
/// Writers read through [`Source`] and confirm provisional allocation ownership before
/// publishing references.
pub(crate) trait Store: Source {
    /// Borrows the session's original image with its effective address context.
    fn opened_source(&self) -> OpenedSource<'_>;

    /// This file's validated address and length field widths.
    fn widths(&self) -> FormatWidths;

    /// Returns the stored-to-physical address mapping for this image.
    fn addresses(&self) -> AddressContext;

    /// Returns the absolute end-of-file address stored in the superblock.
    fn logical_end(&self) -> u64;

    /// Requires complete, disjoint storage ownership and a reachable target header.
    ///
    /// The editor may cache the ownership analysis while checked allocations preserve it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if ownership analysis is incomplete or `header` is
    /// unreachable. Propagates storage-analysis errors, including invalid overlapping extents.
    fn require_exclusive_chunk_storage(&self, header: ObjectId) -> Result<(), Error>;

    /// Consumes a confirmed predecessor retirement under the session's allocation policy.
    ///
    /// The editor applies strategy admission, threshold, and page-class rules. For
    /// persistent sessions, the editor persists changed ownership during the next manager
    /// publication.
    fn retire(&mut self, retirement: RetiredChunk);

    /// Returns the editing session's allocation publication state.
    fn allocation_publication(&self) -> &AllocationPublication;

    /// Publishes allocation ownership and any required EOF change before references.
    ///
    /// The returned confirmation is bound to this session and allocation generation.
    /// In persistent sessions, the editor writes replacement managers excluding complete
    /// allocations.
    /// The storage layer completes its content and publication barriers before returning.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier outcome is uncertain.
    /// Propagates metadata read, parse, placement, write, and synchronization errors.
    fn publish_allocations(&mut self) -> Result<ConfirmedPublication, Error>;

    /// Writes `bytes` into provisional storage of `class`.
    ///
    /// Chunk payloads use [`AllocationClass::Raw`]. Extensible-array data and super blocks use
    /// [`AllocationClass::Metadata`]. In PAGE files, the editor places chunk payloads in raw pages
    /// and reserves complete metadata blocks in metadata pages.
    /// During reclamation, the editor returns index storage to raw-page free space only where
    /// its PAGE placement is proven.
    ///
    /// The store may reuse committed free space or append at end-of-file. Nonpersistent sessions
    /// can also allocate from classic aggregates. The complete allocation lies within the image's
    /// logical length on return. The editor checks persistent manager authority before mutation.
    ///
    /// Before publishing a reference, the caller obtains confirmation through
    /// [`publish_allocations`](Self::publish_allocations) and
    /// [`ConfirmedPublication::confirm`]. A synchronization barrier alone grants no permission.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier outcome is uncertain.
    /// Returns [`Error::Format`] if address conversion or PAGE allocation geometry is invalid.
    /// Returns [`Error::Io`] if writing the allocation fails.
    /// Propagates authoritative-manager read and parse errors.
    fn allocate(
        &mut self,
        bytes: &[u8],
        class: AllocationClass,
    ) -> Result<ProvisionalAllocation, Error>;
    /// Reserves storage and encodes its bytes for the chosen physical address.
    ///
    /// Calls `prepare` once and requires its result to contain exactly `length` bytes. The
    /// allocation and publication rules match [`Self::allocate`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the encoded length differs from the reservation.
    /// Propagates preparation and allocation errors.
    fn allocate_encoded<B: AsRef<[u8]>>(
        &mut self,
        length: u64,
        class: AllocationClass,
        prepare: impl FnOnce(PhysicalAddress) -> Result<B, Error>,
    ) -> Result<ProvisionalAllocation, Error>;

    /// Allocates missing Fixed Array storage while retaining any existing header geometry.
    ///
    /// The store selects combined or separate placement. The caller confirms the returned
    /// initialization allocations before publishing the header or data-block reference.
    ///
    /// # Errors
    ///
    /// Propagates geometry, reservation, emission, and provisioning errors.
    fn allocate_fixed_initialization(
        &mut self,
        plan: &FixedArrayPlan,
        header: Option<FixedArrayHeader>,
    ) -> Result<FixedInitialization, Error>
    where
        Self: Sized,
    {
        FixedInitialization::allocate(self, plan, header)
    }

    /// Allocates empty Extensible Array storage as combined or separately placed blocks.
    ///
    /// Returns the header allocation and, for separate placement, an index-block allocation.
    /// The caller publishes ownership and confirms every returned token before linking the header.
    ///
    /// # Errors
    ///
    /// Propagates geometry, placement, encoding, allocation, and I/O errors.
    fn allocate_ea_initialization(
        &mut self,
        initialization: &ExtensibleArrayInitialization,
    ) -> Result<(ProvisionalAllocation, Option<ProvisionalAllocation>), Error> {
        let addresses = self.addresses();
        self.allocate_encoded(
            initialization.allocation_len(),
            AllocationClass::Metadata,
            |address| {
                initialization
                    .encode_at(addresses.relative(address)?)
                    .map_err(Error::Format)
            },
        )
        .map(|allocation| (allocation, None))
    }

    /// Reserves provisional storage for ordered emission at the chosen physical address.
    ///
    /// After reserving storage, calls `emit` once with the address and a slice-writing callback.
    /// The producer must supply exactly `length` bytes across its slices. Empty slices are
    /// accepted. The sink rejects an oversized slice before writing it and rejects subsequent
    /// writes after a failure, including failures the producer catches.
    ///
    /// The allocation and publication rules match [`Self::allocate`]. The complete allocation
    /// lies within the image on successful return. A producer or sink failure may leave partial
    /// unpublished bytes. The editor applies its allocator restoration policy on failure.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::SerializationError`] for a short emission or an oversized slice,
    /// or [`FormatError::Internal`] if the producer reports success after a rejected sink write.
    /// Propagates reservation, producer, placement, I/O, and allocator restoration errors.
    fn allocate_emitted(
        &mut self,
        length: NonZeroByteLength,
        class: AllocationClass,
        emit: impl FnOnce(
            PhysicalAddress,
            &mut dyn FnMut(&[u8]) -> Result<(), Error>,
        ) -> Result<(), Error>,
    ) -> Result<ProvisionalAllocation, Error>;

    /// Marks allocation publication uncertain after an unconfirmed outcome.
    fn make_allocation_uncertain(&mut self);

    /// Writes reference metadata after every pending allocation has been confirmed.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if allocations remain provisional, or
    /// [`Error::AllocationPublicationUncertain`] if an earlier outcome is uncertain.
    /// Propagates errors from writing or draining buffered ranges.
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error>;
    /// An ordering barrier: everything written before this reaches the operating
    /// system before anything written after it, and under
    /// [`SyncPolicy::Always`](crate::SyncPolicy) it is forced to durable storage
    /// as well.
    ///
    /// [`SyncPolicy::OnClose`](crate::SyncPolicy) drains gathered writes, including explicit
    /// page buffers, without requesting durability.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if allocations remain provisional, or
    /// [`Error::AllocationPublicationUncertain`] if an earlier outcome is uncertain.
    /// Propagates errors from writing or synchronizing buffered ranges.
    fn sync(&mut self) -> Result<(), Error>;

    /// Read an offset-sized address at `offset`.
    fn read_addr_at(&self, offset: u64) -> Result<StoredAddress, Error> {
        let mut buf = [0u8; 8];
        let width = usize::from(self.widths().offsets.get());
        self.read_at(offset, &mut buf[..width])?;
        Ok(StoredAddress::new(u64::from_le_bytes(buf)))
    }

    /// Writes a changed field and its checksum together, starting at the field's offset.
    ///
    /// Reads `[start, cks_off + 4)`, copies `value` at `at`, and recomputes the Jenkins checksum
    /// over the body preceding `cks_off`. Passes the suffix from `at` through the checksum to
    /// [`Self::write_at`] in one call, including when the suffix crosses gather-page boundaries.
    /// The write may tear at a device boundary.
    ///
    /// The read covers the whole body for checksum computation. The write covers only the
    /// suffix, bounding write amplification when updating an element near the end of a block.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if the field extends outside the checksummed body.
    /// Returns [`FormatError::OffsetOverflow`] if the complete physical range through the checksum
    /// exceeds [`u64::MAX`], before reading or writing. Propagates conversion, read, and write
    /// errors.
    fn publish_checksummed(
        &mut self,
        start: u64,
        cks_off: u64,
        at: u64,
        value: &[u8],
    ) -> Result<(), Error> {
        let body_length = cks_off.checked_sub(start).ok_or(Error::AppendUnsupported(
            "a checksummed structure was patched outside itself",
        ))?;
        let span = FileRange::new(
            PhysicalAddress::new(start),
            body_length
                .checked_add(EA_CHECKSUM_SIZE)
                .ok_or(FormatError::OffsetOverflow {
                    offset: cks_off,
                    length: EA_CHECKSUM_SIZE,
                })?,
        )?
        .length()
        .to_usize()?;
        let body_length = body_length.to_usize()?;
        // A range outside the structure is refused rather than asserted: every
        // offset here is derived from geometry parsed out of the file, so a
        // malformed one is untrusted input, not a broken invariant. Checked
        // before the read, so a refusal costs nothing.
        let from = at
            .checked_sub(start)
            .and_then(|offset| offset.to_usize().ok())
            .filter(|&offset| {
                offset
                    .checked_add(value.len())
                    .is_some_and(|end| end <= body_length)
            })
            .ok_or(Error::AppendUnsupported(
                "a checksummed structure was patched outside itself",
            ))?;
        let mut bytes = self.read_exact_at(start, span)?;
        bytes[from..from + value.len()].copy_from_slice(value);
        let (body, cks_field) = bytes.split_at_mut(body_length);
        cks_field.copy_from_slice(&checksum::jenkins_lookup3(body).to_le_bytes());
        self.write_at(at, &bytes[from..])
    }
}

/// Widest Extensible-Array element this engine can be asked to place: an
/// 8-byte address, a stored size no wider than the `u64` that fills it, and a
/// 4-byte filter mask. `locate` refuses any file naming more.
const MAX_EA_ELEM: usize = 8 + 8 + 4;

/// Absolute file offsets of the object-header messages a caller may need to
/// parse after locating a dataset.
pub(crate) struct MessageSpans {
    /// `(data_off, size)` of the Filter Pipeline message, when present.
    pub filter: Option<(u64, usize)>,
    /// The Fill Value message, when present: its type (which distinguishes the
    /// versioned form from the legacy one) and `(data_off, size)`. An append
    /// needs it because the overhang of the chunk it completes must hold the
    /// dataset's fill value rather than zeros (issue #296).
    pub fill: Option<(MessageType, u64, usize)>,
}

/// Pairs parsed datatype and checked append geometry with remaining message locations.
pub(crate) struct LocateResult {
    pub datatype: Datatype,
    pub located: AppendIndex,
    pub spans: MessageSpans,
}

/// Checked append geometry for an unlimited Extensible Array or a supported finite index.
pub(crate) enum AppendIndex {
    /// The index header and index block are allocated.
    Allocated(Located),
    /// A finite dataset indexed by a retained or deferred Fixed Array.
    Fixed(Box<FixedIndex>),
    /// A finite dataset whose layout message stores its chunk reference directly.
    Single(SingleIndex),
    /// The empty dataset's index-header address is undefined.
    Unallocated(UnallocatedIndex),
}

impl AppendIndex {
    /// Returns allocated Extensible Array metadata by value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for a finite index or an unallocated Extensible Array.
    #[cfg(test)]
    fn into_allocated(self) -> Result<Located, Error> {
        match self {
            Self::Allocated(located) => Ok(located),
            Self::Fixed(_) | Self::Single(_) | Self::Unallocated(_) => {
                Err(Error::AppendUnsupported(
                    "the dataset's extensible-array index is not allocated; use Dataset::append_staged",
                ))
            }
        }
    }

    /// Borrows allocated Extensible Array metadata.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for a finite index or an unallocated Extensible Array.
    #[cfg(test)]
    pub(crate) fn allocated(&self) -> Result<&Located, Error> {
        match self {
            Self::Allocated(located) => Ok(located),
            Self::Fixed(_) | Self::Single(_) | Self::Unallocated(_) => {
                Err(Error::AppendUnsupported(
                    "the dataset's extensible-array index is not allocated; use Dataset::append_staged",
                ))
            }
        }
    }

    /// Borrows allocated Extensible Array metadata for mutation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for a finite index or an unallocated Extensible Array.
    pub(crate) fn allocated_mut(&mut self) -> Result<&mut Located, Error> {
        match self {
            Self::Allocated(located) => Ok(located),
            Self::Fixed(_) | Self::Single(_) | Self::Unallocated(_) => {
                Err(Error::AppendUnsupported(
                    "the dataset's extensible-array index is not allocated; use Dataset::append_staged",
                ))
            }
        }
    }

    /// Returns the number of dataset elements in a complete chunk.
    pub(crate) fn chunk_elems(&self) -> u64 {
        match self {
            Self::Allocated(located) => located.chunk_elems,
            Self::Unallocated(index) => index.chunk_elems,
            Self::Fixed(index) => index.geometry.chunk_elems,
            Self::Single(index) => index.geometry.chunk_elems,
        }
    }

    /// Returns the uncompressed size of a complete chunk in bytes.
    pub(crate) fn chunk_bytes(&self) -> usize {
        match self {
            Self::Allocated(located) => located.chunk_bytes,
            Self::Unallocated(index) => index.chunk_bytes,
            Self::Fixed(index) => index.geometry.chunk_bytes,
            Self::Single(index) => index.geometry.chunk_bytes,
        }
    }

    /// Returns the size of one dataset element in bytes.
    pub(crate) fn element_size(&self) -> NonZeroUsize {
        match self {
            Self::Allocated(located) => located.elem_bytes,
            Self::Unallocated(index) => index.element_size,
            Self::Fixed(index) => index.geometry.element_size,
            Self::Single(index) => index.geometry.element_size,
        }
    }

    /// Returns the current extent along the first axis.
    ///
    /// An unallocated Extensible Array has zero current elements.
    pub(crate) fn current_dim(&self) -> u64 {
        match self {
            Self::Allocated(located) => located.current_dim,
            Self::Unallocated(_) => 0,
            Self::Fixed(index) => index.geometry.current_dim,
            Self::Single(index) => index.geometry.current_dim,
        }
    }

    /// Returns the first dimension's maximum extent for append preflight.
    pub(crate) fn maximum(&self) -> MaxExtent {
        match self {
            Self::Allocated(_) | Self::Unallocated(_) => MaxExtent::Unlimited,
            Self::Fixed(index) => MaxExtent::Fixed(index.geometry.maximum),
            Self::Single(index) => MaxExtent::Fixed(index.geometry.maximum),
        }
    }

    /// Returns the chunk-layout flags used for partial-edge filtering.
    pub(crate) fn flags(&self) -> ChunkedLayoutFlags {
        match self {
            Self::Allocated(located) => located.flags,
            Self::Unallocated(index) => index.flags,
            Self::Fixed(index) => index.geometry.flags,
            Self::Single(index) => index.geometry.flags,
        }
    }
}

/// Checked rank-one finite extents and chunk size, with the mutable dimension field.
struct FiniteGeometry {
    widths: FormatWidths,
    dimension: ObjectHeaderField,
    current_dim: u64,
    maximum: u64,
    flags: ChunkedLayoutFlags,
    chunk_elems: u64,
    element_size: NonZeroUsize,
    chunk_bytes: usize,
}

/// Checked creation and chunk geometry for an empty Extensible Array.
///
/// The dataset has zero current elements and an undefined index-header address.
pub(crate) struct UnallocatedIndex {
    /// Creation parameters from the dataset's layout message.
    pub(crate) creation: ExtensibleArrayCreationParameters,
    widths: FormatWidths,
    /// Chunk-layout flags for partial-edge filtering.
    pub(crate) flags: ChunkedLayoutFlags,
    /// Elements per chunk along the unlimited axis.
    pub(crate) chunk_elems: u64,
    /// Bytes per dataset element.
    pub(crate) element_size: NonZeroUsize,
    /// Uncompressed bytes in one complete chunk.
    pub(crate) chunk_bytes: usize,
    /// Location of the layout field to update after allocating the index.
    pub(crate) reference: IndexInitializationReference,
}

/// Checked physical ranges of an undefined index-header field and its object-header chunk.
pub(crate) struct IndexInitializationReference {
    header: ObjectHeaderField,
}

impl IndexInitializationReference {
    /// Checks the message-local address field and its containing object-header chunk.
    ///
    /// # Errors
    ///
    /// Returns range and geometry errors from [`ObjectHeaderField::new`].
    fn new(message: &WalkedMessage, field: core::ops::Range<usize>) -> Result<Self, FormatError> {
        Ok(Self {
            header: ObjectHeaderField::new(message, field)?,
        })
    }

    /// Writes the confirmed index-header address and its containing object-header checksum.
    ///
    /// Encodes the address at the store's offset width. The caller synchronizes this reference
    /// before appending through the initialized index.
    ///
    /// # Errors
    ///
    /// Propagates allocation-ownership, defined-address encoding, range, and publication errors.
    pub(crate) fn publish(
        &self,
        file: &mut impl Store,
        allocation: &ConfirmedAllocation,
    ) -> Result<(), Error> {
        let allocation = file.allocation_publication().require(allocation)?;
        let address = file.addresses().relative(allocation.start())?;
        let mut encoded = [0; 8];
        hdf5_pure_format::__private::write_defined_stored_address_at(
            &mut encoded,
            0,
            address,
            file.widths().offsets,
        )?;
        self.header
            .publish(file, &encoded[..usize::from(file.widths().offsets.get())])
    }
}

/// A physical field range within a complete checksummed object-header chunk.
struct ObjectHeaderField {
    field: FileRange,
    chunk: FileRange,
}

impl ObjectHeaderField {
    /// Resolves a message-local field and checks it against the chunk's checksum coverage.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the field is empty, exceeds its
    /// message, or lies outside the chunk body. Returns [`FormatError::OffsetOverflow`] if a
    /// physical range exceeds [`u64::MAX`].
    fn new(message: &WalkedMessage, field: core::ops::Range<usize>) -> Result<Self, FormatError> {
        if field.is_empty() || field.end > message.size {
            return Err(FormatError::InvalidExtensibleArrayGeometry);
        }
        let field = FileRange::new(
            PhysicalAddress::new(message.data_off).checked_add(field.start.to_u64())?,
            field.len().to_u64(),
        )?;
        let chunk = FileRange::new(
            PhysicalAddress::new(message.chunk_start),
            message
                .chunk_msg_end
                .checked_sub(message.chunk_start)
                .and_then(|length| length.checked_add(EA_CHECKSUM_SIZE))
                .ok_or(FormatError::InvalidExtensibleArrayGeometry)?,
        )?;
        if field.start() < chunk.start() || field.end().get() > message.chunk_msg_end {
            return Err(FormatError::InvalidExtensibleArrayGeometry);
        }
        Ok(Self { field, chunk })
    }

    /// Publishes exactly this field's bytes with the containing chunk's updated checksum.
    ///
    /// Checks the complete chunk against published EOF and physical source bounds before I/O.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `bytes` has a different length. Propagates range errors
    /// from [`metadata::require_range`] and publication errors from [`Store::publish_checksummed`].
    fn publish(&self, file: &mut impl Store, bytes: &[u8]) -> Result<(), Error> {
        if bytes.len().to_u64() != self.field.length() {
            return Err(FormatError::Internal(
                "an object-header field has a different encoded length".into(),
            )
            .into());
        }
        metadata::require_range(file, self.chunk)?;
        file.publish_checksummed(
            self.chunk.start().get(),
            self.chunk.end().get() - EA_CHECKSUM_SIZE,
            self.field.start().get(),
            bytes,
        )
    }
}

/// Allocated index geometry and checked message locations maintained across appends.
///
/// Retains the parsed address and length widths for planning and field publication.
pub(crate) struct Located {
    dimension: ObjectHeaderField,
    /// The currently published length along axis 0.
    pub current_dim: u64,
    length_width: LengthWidth,

    /// The flags the chunked layout sets, such as [`DONT_FILTER_PARTIAL_BOUND_CHUNKS`], under which
    /// a partial edge chunk is stored without the filter pipeline.
    ///
    /// [`DONT_FILTER_PARTIAL_BOUND_CHUNKS`]:
    /// hdf5_pure_format::__private::DONT_FILTER_PARTIAL_BOUND_CHUNKS
    pub flags: ChunkedLayoutFlags,
    /// Elements per chunk along axis 0 (the only varying axis for rank 1).
    pub chunk_elems: u64,
    /// Bytes per dataset element, the element size the chunked layout stores.
    pub elem_bytes: NonZeroUsize,
    /// Bytes per chunk (uncompressed).
    pub chunk_bytes: usize,

    /// Extensible Array client id: 0 = unfiltered (bare-address element), 1 =
    /// filtered (`address + compressed_size + filter_mask` element).
    pub client_id: u8,
    /// Extensible Array header address and derived geometry.
    pub ea_addr: StoredAddress,
    pub offset_width: OffsetWidth,
    pub geom: ExtensibleArrayGeometry,
    pub idx_blk_elmts: u64,
    /// Size of one stored EA element in bytes (offset size for unfiltered; wider
    /// for filtered).
    pub ea_elem_size: usize,
    /// The parsed maximum element bit count and data block page exponent.
    pub ea_bits: ExtensibleArrayBitGeometry,
    /// Address of the EA index block (`EAIB`).
    pub index_block_addr: StoredAddress,
    /// Number of chunk positions within the current dataset extent, including unallocated ones.
    pub num_chunks: u64,
    /// Statistics computed from referenced blocks, invalidated before index mutation and
    /// cached after the header publication barrier.
    allocation_stats: Option<ExtensibleArrayStats>,
}

impl Located {
    /// Reads the datatype and checked geometry for an immediate append at `oh_addr`.
    ///
    /// The dataset must be rank one with an unlimited Extensible Array or a finite Single Chunk
    /// or Fixed Array index. An undefined Extensible Array header address is accepted only at an
    /// empty extent.
    /// Finite-index lookup requires complete, disjoint storage ownership and a reachable header.
    /// The caller supplies `unsupported` for policy rejections and decides which filters it
    /// accepts.
    /// The result includes filter and fill message locations for that decision.
    ///
    /// For allocated Extensible Arrays, the reader verifies the header, index block, and every
    /// defined secondary block, including their checksums regardless of the `checksum` feature.
    /// It checks complete index and data block allocations against the published EOF and physical
    /// source, computes allocation statistics, and leaves file storage unchanged. Reads use bounded
    /// [`Source`] windows.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingMessage`] for a missing dataspace, layout, or datatype. Returns
    /// [`Error::Format`] for malformed chunk geometry, inconsistent element sizes, invalid index
    /// geometry, failed checksums, or metadata read and parse errors. Returns
    /// [`Error::InvalidSpaceOwnershipExtent`] for an allocation beyond the published EOF and
    /// [`Error::AppendUnsupported`] for an unallocated index block. Returns `unsupported` for other
    /// unsupported storage or metadata.
    pub(crate) fn locate_at<F: Store>(
        file: &F,
        oh_addr: ObjectId,
        unsupported: fn(&'static str) -> Error,
    ) -> Result<LocateResult, Error> {
        let widths = file.widths();
        let os = widths.offsets.get();

        let walk = walk_v2_object_header(file, oh_addr.header_address(), widths)?;

        let dataspace_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DATASPACE)
            .ok_or(Error::MissingMessage(MessageType::DATASPACE))?;
        let layout_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DATA_LAYOUT)
            .ok_or(Error::MissingMessage(MessageType::DATA_LAYOUT))?;
        let datatype_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DATATYPE)
            .ok_or(Error::MissingMessage(MessageType::DATATYPE))?;
        let filter_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FILTER_PIPELINE);
        // The versioned message wins over the legacy one when a header carries
        // both, matching how the read path picks.
        let fill_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FILL_VALUE)
            .or_else(|| {
                walk.messages
                    .iter()
                    .find(|m| m.msg_type == MessageType::FILL_VALUE_OLD)
            });

        // Each of these is parsed below as though its bytes were the message. A
        // *shared* record's body is a reference to a message stored elsewhere —
        // the shape `H5Tcommit` gives a dataset's element type — and decodes as a
        // well-formed type of the wrong kind, so an append against it would size
        // its elements from a type the dataset does not have. Refuse instead.
        for msg in [Some(dataspace_msg), Some(datatype_msg), filter_msg]
            .into_iter()
            .flatten()
        {
            if msg.flags.is_shared() {
                return Err(unsupported(
                    "dataset has a committed (shared) datatype, dataspace, or filter pipeline",
                ));
            }
        }

        // Read the rank-one current and maximum extents before selecting the index writer.
        let ds_bytes = file.read_metadata_at(dataspace_msg.data_off, dataspace_msg.size)?;
        let (dataspace, dimension_fields) =
            Dataspace::parse_with_dimension_fields(&ds_bytes, widths.lengths)?;
        if dataspace.rank != 1 {
            return Err(unsupported("only rank-1 datasets are supported"));
        }
        let current_dim = dataspace.dimensions[0];
        let maximum = dataspace
            .max_dimensions
            .as_deref()
            .map_or(MaxExtent::Fixed(current_dim), |maximum| maximum[0]);
        let dimension = ObjectHeaderField::new(dataspace_msg, dimension_fields)?;
        metadata::require_range(file, dimension.chunk)?;

        let layout_bytes = file.read_metadata_at(layout_msg.data_off, layout_msg.size)?;
        let (layout, fields) = DataLayout::parse_with_index_fields(&layout_bytes, widths)?;
        if maximum != MaxExtent::Unlimited
            && !matches!(
                fields,
                Some(ChunkIndexFields::SingleChunk(_) | ChunkIndexFields::FixedArray { .. })
            )
        {
            return Err(unsupported("dataset has no unlimited (maxshape) dimension"));
        }
        let dt_bytes = file.read_metadata_at(datatype_msg.data_off, datatype_msg.size)?;
        let (datatype, _) = hdf5_pure_format::__private::parse_datatype(&dt_bytes)?;
        crate::file_writer::require_reference_width(widths, &datatype)?;
        let element_size =
            hdf5_pure_format::__private::stored_element_size_usize(&datatype, widths.offsets)?;
        if let DataLayout::Chunked {
            chunk_dimensions, ..
        } = &layout
            && let Some(&stored_size) = chunk_dimensions.last()
            && stored_size != element_size.get() as u64
        {
            return Err(FormatError::ChunkElementSizeMismatch {
                datatype: element_size.get(),
                layout: stored_size.to_usize()?,
            }
            .into());
        }
        let ChunkedGeometry {
            spatial,
            element_size: elem_bytes,
            raw_size,
            ..
        } = ChunkedGeometry::with_offset_width(widths.offsets, &datatype, &dataspace, &layout)?;
        let chunk_elems = spatial[0];
        let chunk_bytes = if matches!(
            fields,
            Some(ChunkIndexFields::SingleChunk(_) | ChunkIndexFields::FixedArray { .. })
        ) {
            let length = chunked_read::bounded_chunk_len(spatial, elem_bytes)?;
            if length > isize::MAX as usize {
                return Err(FormatError::ValueTooLargeForPlatform {
                    value: raw_size,
                    target: "single-chunk allocation",
                }
                .into());
            }
            length
        } else {
            raw_size.to_usize()?
        };

        let spans = MessageSpans {
            filter: filter_msg.map(|m| (m.data_off, m.size)),
            fill: fill_msg.map(|m| (m.msg_type, m.data_off, m.size)),
        };
        if let (
            DataLayout::Chunked {
                index: ChunkIndexLayout::FixedArray { address },
                flags,
                ..
            },
            Some(ChunkIndexFields::FixedArray {
                page_bits,
                address: field,
            }),
        ) = (&layout, &fields)
        {
            let MaxExtent::Fixed(maximum) = maximum else {
                return Err(unsupported("a fixed-array index requires a finite maximum"));
            };
            return Ok(LocateResult {
                datatype,
                located: AppendIndex::Fixed(Box::new(FixedIndex::new(
                    file,
                    oh_addr,
                    FiniteGeometry {
                        widths,
                        dimension,
                        current_dim,
                        maximum,
                        flags: *flags,
                        chunk_elems,
                        element_size: elem_bytes,
                        chunk_bytes,
                    },
                    layout_msg,
                    field.clone(),
                    *address,
                    *page_bits,
                    filter_msg.is_some(),
                )?)),
                spans,
            });
        }
        if let (
            DataLayout::Chunked {
                index: ChunkIndexLayout::SingleChunk { address, filtered },
                flags,
                ..
            },
            Some(ChunkIndexFields::SingleChunk(field)),
        ) = (&layout, &fields)
        {
            let MaxExtent::Fixed(maximum) = maximum else {
                return Err(unsupported(
                    "a single-chunk index requires a finite maximum",
                ));
            };
            if filtered.is_some() != filter_msg.is_some() {
                return Err(unsupported("single-chunk filter metadata is inconsistent"));
            }
            let record = address.map(|address| ChunkRecord {
                address,
                stored_size: filtered.map_or(raw_size, |filtered| filtered.filtered_size),
                filter_mask: filtered.map_or(0, |filtered| filtered.filter_mask),
            });
            return Ok(LocateResult {
                datatype,
                located: AppendIndex::Single(SingleIndex::new(
                    file,
                    oh_addr,
                    FiniteGeometry {
                        widths,
                        dimension,
                        current_dim,
                        maximum,
                        flags: *flags,
                        chunk_elems,
                        element_size: elem_bytes,
                        chunk_bytes,
                    },
                    layout_msg,
                    field.clone(),
                    &layout_bytes,
                    record,
                )?),
                spans,
            });
        }
        let (ea_addr, creation, flags, layout_address) = match (&layout, fields) {
            (
                DataLayout::Chunked {
                    index: ChunkIndexLayout::ExtensibleArray { address, creation },
                    flags,
                    ..
                },
                Some(ChunkIndexFields::ExtensibleArray(field)),
            ) => (*address, *creation, *flags, field.range()),
            _ => {
                return Err(unsupported(
                    "only Extensible-Array-indexed chunked datasets are supported",
                ));
            }
        };
        let Some(ea_addr) = ea_addr else {
            if current_dim != 0 {
                return Err(unsupported(
                    "an unallocated extensible-array index requires an empty dataset",
                ));
            }
            return Ok(LocateResult {
                datatype,
                located: AppendIndex::Unallocated(UnallocatedIndex {
                    creation,
                    widths,
                    flags,
                    chunk_elems,
                    element_size: elem_bytes,
                    chunk_bytes,
                    reference: {
                        let reference =
                            IndexInitializationReference::new(layout_msg, layout_address)?;
                        metadata::require_range(file, reference.header.chunk)?;
                        reference
                    },
                }),
                spans,
            });
        };

        let array = metadata::read_header(file, ea_addr)?;
        let ea_header = array.header();
        let has_filters = filter_msg.is_some();
        if (ea_header.client_id == 1) != has_filters {
            return Err(unsupported(
                "dataset filter metadata is inconsistent (chunk-index client id \
                 disagrees with the filter pipeline)",
            ));
        }
        // Immediate append writes a stored-size field of one to eight bytes.
        // The header parser accepts an empty size field as well.
        let elem_w = ea_header.element_size as usize;
        if ea_header.client_id == 1 && !(os as usize + 5..=os as usize + 12).contains(&elem_w) {
            return Err(unsupported(
                "malformed filtered extensible-array element width",
            ));
        }
        if ea_header.client_id == EA_CLIENT_ID_UNFILTERED && elem_w != os as usize {
            return Err(unsupported(
                "malformed unfiltered extensible-array element width",
            ));
        }
        let geom = array.geometry()?;
        let index_block_addr = ea_header.index_block_address;
        // The dataspace dimension is the single commit point; the EA element
        // count is published one step earlier. If a prior writer crashed between
        // the two, the on-disk EA count is ahead of the committed dimension. Seed
        // the chunk count from the *committed* dimension so the next append rolls
        // forward from the last commit -- overwriting any slots a crashed writer
        // wrote but never committed -- instead of appending past them.
        let num_chunks = current_dim.div_ceil(chunk_elems);

        let mut located = Located {
            dimension,
            current_dim,
            length_width: widths.lengths,
            flags,
            chunk_elems,
            elem_bytes,
            chunk_bytes,
            client_id: ea_header.client_id,
            ea_addr,
            offset_width: widths.offsets,
            geom,
            idx_blk_elmts: ea_header.idx_blk_elmts as u64,
            ea_elem_size: ea_header.element_size as usize,
            ea_bits: ea_header.bit_geometry(),
            index_block_addr,
            num_chunks,
            allocation_stats: None,
        };
        located.allocation_stats = Some(located.read_allocation_stats(file, &array)?);
        Ok(LocateResult {
            datatype,
            located: AppendIndex::Allocated(located),
            spans,
        })
    }

    /// Returns the physical position of the first current-dimension field.
    #[cfg(test)]
    pub(crate) fn dimension_offset(&self) -> u64 {
        self.dimension.field.start().get()
    }

    /// Reads the live partial tail and captures its index reference and complete allocation.
    ///
    /// The chunk is decoded as a reader at the current extent decodes it, through the filter
    /// pipeline unless the layout flags exempt it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if the index has no record for the chunk. Propagates
    /// errors from reading the record, capturing its allocation, or
    /// [`ChunkInfo::read_live_elements`].
    fn read_live_tail<F: Store>(
        &self,
        file: &F,
        scratch: &mut FilterScratch,
        pipeline: Option<&FilterPipeline>,
        ctx: ChunkContext<'_>,
    ) -> Result<(Vec<u8>, CapturedChunkReference), Error> {
        let rec = self
            .read_element(file, self.current_dim / self.chunk_elems)?
            .ok_or(Error::AppendUnsupported(
                "trailing partial chunk is missing from the index",
            ))?;
        let chunk = self.chunk_info(rec);
        let allocation = ChunkAllocation::capture(file.addresses(), &chunk)?;
        let live = chunk.read_live_elements(
            &PhysicalSource::new(&file.opened_source()).stored(),
            scratch,
            self.flags,
            pipeline,
            ctx,
            self.current_dim,
        )?;
        Ok((
            live,
            CapturedChunkReference {
                owner: ChunkReferenceOwner::new(file, self),
                slot: self.current_dim / self.chunk_elems,
                dimension: self.current_dim,
                record: rec,
                allocation,
            },
        ))
    }

    /// Preserves complete allocation lengths when adapting an index record for chunk decoding.
    ///
    /// Uses the checked full chunk size for unfiltered chunks. Uses the record's stored size for
    /// filtered chunks, including filter-exempt payloads.
    fn chunk_info(&self, rec: ElemRecord) -> ChunkInfo {
        let stored_size = if self.client_id == EA_CLIENT_ID_UNFILTERED {
            self.chunk_bytes.to_u64()
        } else {
            rec.stored_size
        };
        ChunkInfo {
            chunk_size: StoredChunkSize::v4(stored_size),
            filter_mask: rec.filter_mask,
            offsets: vec![self.current_dim - self.current_dim % self.chunk_elems],
            address: rec.addr,
        }
    }

    /// Read the element record stored at byte offset `off`.
    fn read_element_at<F: Store>(&self, file: &F, off: u64) -> Result<ElemRecord, Error> {
        let os = file.widths().offsets.get() as usize;
        if self.client_id == 0 {
            return Ok(ElemRecord::addr_only(file.read_addr_at(off)?));
        }
        // One bounded read of the whole fixed-width element record.
        let elem = file.read_exact_at(off, self.ea_elem_size)?;
        let addr_w = os;
        let mut a = [0u8; 8];
        a[..addr_w].copy_from_slice(&elem[..addr_w]);
        let addr = StoredAddress::new(u64::from_le_bytes(a));
        let csz = self.ea_elem_size - os - 4;
        let mut stored_size = 0u64;
        for i in 0..csz {
            stored_size |= (elem[os + i] as u64) << (8 * i);
        }
        let fm = os + csz;
        let filter_mask = u32::from_le_bytes([elem[fm], elem[fm + 1], elem[fm + 2], elem[fm + 3]]);
        Ok(ElemRecord {
            addr,
            stored_size,
            filter_mask,
        })
    }

    /// Byte offset of element `e`'s record, or `None` when the containing block
    /// (or super block) is not yet allocated, i.e. the element does not exist on
    /// disk yet.
    fn elem_slot_off<F: Store>(&self, file: &F, e: u64) -> Result<Option<u64>, Error> {
        let os = file.widths().offsets.get() as usize;
        let elem_size = self.ea_elem_size as u64;
        let idx = self.idx_blk_elmts;
        let blk_off = self.ea_bits.block_offset_width();
        let page_nelmts = self.ea_bits.page_nelmts().get();

        if e < idx {
            let ib_prefix = (4 + 1 + 1 + os) as u64;
            return Ok(Some(
                self.index_block_addr.get() + ib_prefix + e * elem_size,
            ));
        }

        let region = locate_data_block(&self.geom, idx, e)?;
        let is_paged = region.blocks(page_nelmts).is_paged();
        let slot = e - region.db_start;

        // Resolve the owning super block (if any) and the data-block pointer,
        // returning None if either is unallocated.
        let sblk_addr = match region.parent {
            Parent::Super { sblk_j, .. } => {
                let a = self.super_block_addr(file, sblk_j)?;
                if a.is_undefined(file.widths().offsets) {
                    return Ok(None);
                }
                Some(a)
            }
            Parent::IndexDirect { .. } => None,
        };
        let dblk_ptr_off = self.dblk_ptr_off(sblk_addr, &region, os, blk_off)?;
        let dblk_addr = file.read_addr_at(dblk_ptr_off)?;
        if dblk_addr.is_undefined(file.widths().offsets) {
            return Ok(None);
        }

        let off = if !is_paged {
            let db_prefix = (4 + 1 + 1 + os + blk_off.bytes()) as u64;
            dblk_addr.get() + db_prefix + slot * elem_size
        } else {
            let header_size = (4 + 1 + 1 + os + blk_off.bytes() + 4) as u64;
            let page = slot / page_nelmts;
            let slot_in_page = slot % page_nelmts;
            let page_bytes = page_nelmts * elem_size + 4;
            let page_off = dblk_addr.get() + header_size + page * page_bytes;
            page_off + slot_in_page * elem_size
        };
        Ok(Some(off))
    }

    /// File offset of the data-block-pointer slot for `region` (either a direct
    /// pointer in the index block or a pointer inside the resolved super block).
    fn dblk_ptr_off(
        &self,
        sblk_addr: Option<StoredAddress>,
        region: &DataBlockLoc,
        os: usize,
        blk_off: BlockOffsetWidth,
    ) -> Result<u64, Error> {
        match region.parent {
            Parent::IndexDirect { ordinal } => {
                let ib_prefix = (4 + 1 + 1 + os) as u64;
                Ok(self.index_block_addr.get()
                    + ib_prefix
                    + self.idx_blk_elmts * self.ea_elem_size as u64
                    + (ordinal * os) as u64)
            }
            Parent::Super { dblk_local, .. } => {
                let sblk_addr = sblk_addr.expect("super-block address resolved for a Super parent");
                Ok(sb_dblk_slot_off(
                    os,
                    sblk_addr,
                    dblk_local,
                    region.super_block(self.ea_bits.page_nelmts().get()),
                    blk_off,
                ))
            }
        }
    }

    /// Read element `e`, or `None` if its slot is not allocated / is undefined.
    pub(crate) fn read_element<F: Store>(
        &self,
        file: &F,
        e: u64,
    ) -> Result<Option<ElemRecord>, Error> {
        match self.elem_slot_off(file, e)? {
            None => Ok(None),
            Some(off) => {
                let rec = self.read_element_at(file, off)?;
                if rec.addr.is_undefined(file.widths().offsets) {
                    Ok(None)
                } else {
                    Ok(Some(rec))
                }
            }
        }
    }

    /// Writes `rec` into element slot `e`, allocating missing data or secondary blocks.
    ///
    /// Reuses existing blocks for element updates and recomputes the checksum of the block
    /// or page that contains the element. Invalidates cached allocation statistics before
    /// modifying the index. Planning a subsequent append reads block pointers again until
    /// the header publication barrier completes successfully.
    ///
    /// Before setting a page's initialization bit, drains its buffered contents through
    /// [`Store::sync`]. Under [`SyncPolicy::Always`](crate::SyncPolicy::Always), also requests
    /// durability for those contents.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if an element range exceeds
    /// [`u64::MAX`], or [`FormatError::ValueTooLargeForPlatform`] if a block size or element count
    /// exceeds [`usize::MAX`]. Returns [`Error::AppendUnsupported`] if the
    /// geometry does not cover `e`, the stored chunk size exceeds its field width, or a
    /// checksum publication range lies outside its block. Propagates errors from reading,
    /// encoding, allocating, writing, or synchronizing blocks.
    fn ea_insert<F: Store>(
        &mut self,
        file: &mut F,
        e: u64,
        rec: ConfirmedChunk,
    ) -> Result<(), Error> {
        let rec = rec.record(file)?;
        self.allocation_stats = None;
        let os = file.widths().offsets.get() as usize;
        let elem_size = self.ea_elem_size as u64;
        let idx = self.idx_blk_elmts;
        let blk_off = self.ea_bits.block_offset_width();

        // Inline element slots live directly in the index block.
        if e < idx {
            let ib_prefix = (4 + 1 + 1 + os) as u64;
            let slot_off = self.index_block_addr.get() + ib_prefix + e * elem_size;
            let mut buf = [0u8; MAX_EA_ELEM];
            let n = self.element_bytes(&mut buf, rec)?;
            return self.publish_index_block(file, slot_off, &buf[..n]);
        }

        let region = locate_data_block(&self.geom, idx, e)?;
        let dblk_nelmts = region.dblk_nelmts;
        let sb_geom = region.super_block(self.ea_bits.page_nelmts().get());
        let is_paged = sb_geom.blocks.is_paged();
        let slot = e - region.db_start;
        let block_offset_rel = region.db_start - idx;

        // Ensure the owning super block exists (idempotent) for a Super parent.
        let sblk_addr = match region.parent {
            Parent::Super { sblk_j, .. } => {
                Some(self.ensure_super_block(file, sblk_j, region.sb_block_offset, sb_geom)?)
            }
            Parent::IndexDirect { .. } => None,
        };

        // Resolve the data-block address, allocating a fresh block when the
        // parent pointer is undefined (first touch of the block). Keying on the
        // pointer being undefined -- rather than on `slot == 0` -- makes an
        // in-place element update reuse the existing block instead of leaking it.
        let dblk_ptr_off = self.dblk_ptr_off(sblk_addr, &region, os, blk_off)?;
        let existing = file.read_addr_at(dblk_ptr_off)?;
        let dblk_addr = if existing.is_undefined(file.widths().offsets) {
            let new_addr = self.alloc_undef_data_block(file, dblk_nelmts, block_offset_rel)?;
            #[cfg(test)]
            alloc_probe::note_data_block();
            // As in `ensure_super_block`: an appended block sits above its parent
            // pointer, so they need a barrier between them or an address-ordered
            // flush names a block that is not there yet.
            let publication = file.publish_allocations()?;
            let allocation = publication.confirm(new_addr)?;
            let range = file.allocation_publication().require(&allocation)?;
            let new_addr = file.addresses().relative(range.start())?;
            match region.parent {
                Parent::IndexDirect { .. } => {
                    self.publish_index_block(
                        file,
                        dblk_ptr_off,
                        &self.address_bytes(new_addr)?[..os],
                    )?;
                }
                Parent::Super { .. } => self.publish_super_block(
                    file,
                    sblk_addr.unwrap(),
                    sb_geom,
                    blk_off,
                    dblk_ptr_off,
                    &self.address_bytes(new_addr)?[..os],
                )?,
            }
            new_addr
        } else {
            existing
        };

        if !is_paged {
            let db_prefix = (4 + 1 + 1 + os + blk_off.bytes()) as u64;
            let block_start = dblk_addr.get();
            let elem_off = block_start + db_prefix + slot * elem_size;
            let mut buf = [0u8; MAX_EA_ELEM];
            let n = self.element_bytes(&mut buf, rec)?;
            let cks_off = block_start + db_prefix + dblk_nelmts * elem_size;
            file.publish_checksummed(block_start, cks_off, elem_off, &buf[..n])?;
        } else {
            let page_nelmts = self.ea_bits.page_nelmts().get();
            let header_size = (4 + 1 + 1 + os + blk_off.bytes() + 4) as u64;
            let page = slot / page_nelmts;
            let slot_in_page = slot % page_nelmts;
            let page_bytes = page_nelmts * elem_size + 4;
            let page_off = dblk_addr.get() + header_size + page * page_bytes;
            let elem_off = page_off + slot_in_page * elem_size;
            let mut buf = [0u8; MAX_EA_ELEM];
            let n = self.element_bytes(&mut buf, rec)?;
            let page_cks_off = page_off + page_nelmts * elem_size;
            file.publish_checksummed(page_off, page_cks_off, elem_off, &buf[..n])?;

            if slot_in_page == 0 {
                file.sync()?;
                let sblk_addr = sblk_addr.unwrap();
                let npages = dblk_nelmts / self.ea_bits.page_nelmts().get();
                if let Parent::Super { dblk_local, .. } = region.parent {
                    let global_page = dblk_local as u64 * npages + page;
                    let (byte, set) = sb_page_bit(file, sblk_addr, blk_off, global_page)?;
                    self.publish_super_block(file, sblk_addr, sb_geom, blk_off, byte, &[set])?;
                }
            }
        }
        Ok(())
    }

    /// Address of an already-allocated super block (`sblk_j`-th super-block
    /// pointer in the index block); the undefined sentinel when not yet allocated.
    fn super_block_addr<F: Store>(&self, file: &F, sblk_j: usize) -> Result<StoredAddress, Error> {
        let os = file.widths().offsets.get() as usize;
        let ib_prefix = (4 + 1 + 1 + os) as u64;
        let ndblk_addrs = self.geom.direct_dblk_nelmts().len();
        let slot_off = self.index_block_addr.get()
            + ib_prefix
            + self.idx_blk_elmts * self.ea_elem_size as u64
            + ((ndblk_addrs + sblk_j) * os) as u64;
        file.read_addr_at(slot_off)
    }

    /// Return the address of super block `sblk_j`, allocating an empty one (all
    /// data-block pointers undefined, plus a zeroed page-init bitmap when its data
    /// blocks are paged) if it does not exist yet.
    fn ensure_super_block<F: Store>(
        &self,
        file: &mut F,
        sblk_j: usize,
        sb_block_offset: u64,
        sb: SuperBlockGeometry,
    ) -> Result<StoredAddress, Error> {
        let existing = self.super_block_addr(file, sblk_j)?;
        if !existing.is_undefined(file.widths().offsets) {
            return Ok(existing);
        }
        let os = file.widths().offsets.get() as usize;
        let ib_prefix = (4 + 1 + 1 + os) as u64;
        let ndblk_addrs = self.geom.direct_dblk_nelmts().len();

        let bitmap = vec![0u8; sb.bitmap_size().to_usize()?];
        let undef = vec![StoredAddress::undefined(file.widths().offsets); sb.ndblks.to_usize()?];
        let super_block = hdf5_pure_format::__private::encode_super_block(
            self.ea_addr,
            sb_block_offset,
            &bitmap,
            &undef,
            self.offset_width,
            self.client_id,
            self.ea_bits.block_offset_width(),
        )?;
        let allocation = file.allocate(&super_block, AllocationClass::Metadata)?;
        #[cfg(test)]
        alloc_probe::note_super_block();
        // The block exists before anything names it. When its bytes were appended
        // they sit above the slot below, which is in the index block near the
        // front, so without this the two are one barrier-free window and a
        // gathering image issues them in *address* order — the pointer first
        // (issue #288). A block drawn from freed space may land either side of
        // the slot, and which it is is not knowable here, so the barrier stands
        // unconditionally. See `Store::allocate`.
        let publication = file.publish_allocations()?;
        let allocation = publication.confirm(allocation)?;
        let range = file.allocation_publication().require(&allocation)?;
        let new_addr = file.addresses().relative(range.start())?;

        let slot_off = self.index_block_addr.get()
            + ib_prefix
            + self.idx_blk_elmts * self.ea_elem_size as u64
            + ((ndblk_addrs + sblk_j) * os) as u64;
        self.publish_index_block(file, slot_off, &self.address_bytes(new_addr)?[..os])?;
        Ok(new_addr)
    }

    /// Allocates a provisional data block (`EADB`) with undefined element addresses.
    ///
    /// A block of more than one page of elements is paged.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the element or page element count
    /// exceeds [`usize::MAX`]. Propagates errors from [`Store::allocate`].
    fn alloc_undef_data_block<F: Store>(
        &self,
        file: &mut F,
        dblk_nelmts: u64,
        block_offset_rel: u64,
    ) -> Result<ProvisionalAllocation, Error> {
        let os = self.offset_width;
        let block = hdf5_pure_format::__private::encode_data_block(
            &IndexSlots::dense(&[]),
            0,
            dblk_nelmts.to_usize()?,
            block_offset_rel,
            self.ea_addr,
            os,
            self.client_id != EA_CLIENT_ID_UNFILTERED,
            self.element_size_width(usize::from(os.get())),
            self.client_id,
            self.ea_bits,
        )?;
        file.allocate(&block, AllocationClass::Metadata)
    }

    /// Width of a filtered element's stored-size field. Zero for an unfiltered
    /// array, whose element is a bare address.
    fn element_size_width(&self, os: usize) -> usize {
        if self.client_id == 0 {
            return 0;
        }
        // `element_size` is one byte out of the array header, so a file can name
        // a width the three fields do not fit in. `locate` is where that byte is
        // refused; by here it is in `os + 5 ..= os + 12`.
        debug_assert!(
            (os + 5..=os + 12).contains(&self.ea_elem_size),
            "locate admitted an element width of {} for a {os}-byte address",
            self.ea_elem_size
        );
        self.ea_elem_size - os - 4
    }

    /// Refuse a stored chunk size the element's size field cannot hold.
    ///
    /// Separate from [`element_size_width`](Self::element_size_width) because
    /// the two are needed at different times: the width when an element is laid
    /// out (phase 2 of the apply), the *check* while the plan is still being
    /// built. Re-encoding a trailing chunk is exactly the case where a stored
    /// size can grow past the width, and a refusal raised in phase 2 would land
    /// after phase 1 had already written the chunk bytes — a leaked blob and a
    /// half-applied append. `plan_ea_append` calls this for every blob it is
    /// about to hand over, so the refusal happens before anything is written.
    fn check_stored_size(&self, os: usize, stored_size: u64) -> Result<(), Error> {
        let csz = self.element_size_width(os);
        if self.client_id != 0 && csz < 8 && stored_size >= (1u64 << (8 * csz)) {
            return Err(Error::AppendUnsupported(
                "recompressed chunk size exceeds the dataset's extensible-array element width",
            ));
        }
        Ok(())
    }

    /// Encodes a defined stored address at the recorded width in a zero-padded buffer.
    ///
    /// Only the prefix of [`Self::offset_width`] bytes is part of the address field.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if the address exceeds the width, or
    /// [`FormatError::DefinedAddressIsUndefined`] if it equals the undefined marker.
    fn address_bytes(&self, address: StoredAddress) -> Result<[u8; 8], FormatError> {
        let mut bytes = [0; 8];
        hdf5_pure_format::__private::write_defined_stored_address_at(
            &mut bytes,
            0,
            address,
            self.offset_width,
        )?;
        Ok(bytes)
    }

    /// Encodes one chunk record at the index's recorded widths and returns the bytes used.
    ///
    /// # Errors
    ///
    /// Returns defined-address encoding errors or the stored-size errors of
    /// [`Self::check_stored_size`].
    fn element_bytes(&self, buf: &mut [u8; MAX_EA_ELEM], rec: ElemRecord) -> Result<usize, Error> {
        let os = usize::from(self.offset_width.get());
        hdf5_pure_format::__private::write_defined_stored_address_at(
            buf,
            0,
            rec.addr,
            self.offset_width,
        )?;
        if self.client_id == 0 {
            return Ok(os);
        }
        self.check_stored_size(os, rec.stored_size)?;
        let csz = self.element_size_width(os);
        buf[os..os + csz].copy_from_slice(&rec.stored_size.to_le_bytes()[..csz]);
        buf[os + csz..os + csz + 4].copy_from_slice(&rec.filter_mask.to_le_bytes());
        Ok(os + csz + 4)
    }

    /// Change the index block and republish it, checksum included, as one write.
    fn publish_index_block<F: Store>(
        &self,
        file: &mut F,
        at: u64,
        value: &[u8],
    ) -> Result<(), Error> {
        let os = file.widths().offsets.get() as usize;
        let ib_prefix = (4 + 1 + 1 + os) as u64;
        let ndblk_addrs = self.geom.direct_dblk_nelmts().len();
        let nsblk_addrs = self.geom.nsblk_addrs();
        let cks_off = self.index_block_addr.get()
            + ib_prefix
            + self.idx_blk_elmts * self.ea_elem_size as u64
            + ((ndblk_addrs + nsblk_addrs) * os) as u64;
        file.publish_checksummed(self.index_block_addr.get(), cks_off, at, value)
    }

    /// Change a super block and republish it, checksum included, as one write.
    fn publish_super_block<F: Store>(
        &self,
        file: &mut F,
        sblk_addr: StoredAddress,
        sb: SuperBlockGeometry,
        blk_off: BlockOffsetWidth,
        at: u64,
        value: &[u8],
    ) -> Result<(), Error> {
        let os = file.widths().offsets.get() as usize;
        let prefix = (4 + 1 + 1 + os + blk_off.bytes()) as u64;
        let block_start = sblk_addr.get();
        let cks_off = block_start + prefix + sb.bitmap_size() + sb.ndblks * os as u64;
        file.publish_checksummed(block_start, cks_off, at, value)
    }

    /// Publishes the planned header statistics and their checksum.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if the statistics lie outside the header, and the
    /// read, conversion, or write errors of [`Store::publish_checksummed`]. Returns
    /// [`FormatError::LengthTooLarge`] if a statistic exceeds the store's length width.
    #[cfg(test)]
    fn update_ea_header<F: Store>(
        &self,
        file: &mut F,
        stats: ExtensibleArrayStats,
    ) -> Result<(), Error> {
        let mut bytes = [0; ExtensibleArrayStats::MAX_ENCODED_FIELDS_LEN];
        let length = stats.encode_fields(&mut bytes, file.widths().lengths)?;
        self.publish_statistics(file, &bytes[..length])
    }

    /// Publishes pre-encoded statistics at their format-defined header range.
    ///
    /// # Errors
    ///
    /// Returns geometry-overflow errors from [`ea_sum`] or publication errors from
    /// [`Store::publish_checksummed`].
    fn publish_statistics(&self, file: &mut impl Store, bytes: &[u8]) -> Result<(), Error> {
        let address = self.ea_addr.get();
        let length = ExtensibleArrayHeader::serialized_size_with_widths(file.widths()).to_u64();
        let checksum = ea_sum([address, length])?
            .checked_sub(EA_CHECKSUM_SIZE)
            .ok_or(FormatError::InvalidExtensibleArrayGeometry)?;
        let start = ea_sum([
            address,
            ExtensibleArrayStats::fields_range(file.widths().lengths).start,
        ])?;
        file.publish_checksummed(address, checksum, start, bytes)
    }

    /// Computes the statistics after inserting chunk records in `first_chunk..num_chunks`.
    ///
    /// Reuses the cached allocation counts when available and adds each missing data or
    /// secondary block once. Counts the full capacity of a paged data block, including pages
    /// that are uninitialized.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if a range, pointer offset,
    /// block size, or statistic exceeds [`u64::MAX`], or
    /// [`FormatError::ValueTooLargeForPlatform`] if a data-block index exceeds [`usize::MAX`].
    /// Returns [`Error::AppendUnsupported`] if the geometry does not cover the range or the
    /// index block is unallocated. Propagates errors from reading block pointers.
    fn planned_allocation_stats(
        &self,
        file: &impl Store,
        first_chunk: u64,
        num_chunks: u64,
    ) -> Result<ExtensibleArrayStats, Error> {
        let mut stats = match self.allocation_stats {
            Some(stats) => stats,
            None => {
                self.read_allocation_stats(file, &metadata::read_header(file, self.ea_addr)?)?
            }
        };
        stats.max_idx_set = num_chunks;
        let mut last_new_secondary = None;
        let mut element = first_chunk.max(self.idx_blk_elmts);
        while element < num_chunks {
            let region = locate_data_block(&self.geom, self.idx_blk_elmts, element)?;
            let secondary = match region.parent {
                Parent::IndexDirect { .. } => None,
                Parent::Super { sblk_j, .. } => {
                    let addr = self.super_block_addr(file, sblk_j)?;
                    if addr.is_undefined(self.offset_width) && last_new_secondary != Some(sblk_j) {
                        self.add_super_block(
                            &mut stats,
                            region.super_block(self.ea_bits.page_nelmts().get()),
                        )?;
                        last_new_secondary = Some(sblk_j);
                    }
                    Some(addr)
                }
            };
            let missing = if secondary.is_some_and(|addr| addr.is_undefined(self.offset_width)) {
                true
            } else {
                file.read_addr_at(self.dblk_ptr_off(
                    secondary,
                    &region,
                    usize::from(self.offset_width.get()),
                    self.ea_bits.block_offset_width(),
                )?)?
                .is_undefined(self.offset_width)
            };
            if missing {
                self.add_data_blocks(&mut stats, region.dblk_nelmts, 1)?;
            }
            element = ea_sum([region.db_start, region.dblk_nelmts])?.min(num_chunks);
        }
        Ok(stats)
    }

    /// Computes allocation statistics from defined data and secondary block pointers.
    ///
    /// Counts all referenced blocks, including blocks beyond the current dataset extent.
    /// Verifies index and secondary block framing and checksums, and checks every complete
    /// allocation against the published EOF and physical source. Data block contents are not read.
    /// Uses the current chunk count for the maximum index and includes inline slots in the
    /// element capacity. The version 0 header statistics are defined in "The Extensible Array
    /// Index" of the [format specification, version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if the index block is unallocated, or
    /// [`FormatError::InvalidExtensibleArrayGeometry`] if a pointer offset, block size, or
    /// statistic exceeds [`u64::MAX`]. Returns [`Error::InvalidSpaceOwnershipExtent`] if an
    /// allocation ends past the published EOF, or [`FormatError::UnexpectedSourceEof`] if it ends
    /// past the physical source. Propagates metadata read, parse, and checksum errors.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
    fn read_allocation_stats(
        &self,
        file: &impl Store,
        array: &ExtensibleArray,
    ) -> Result<ExtensibleArrayStats, Error> {
        if self.index_block_addr.is_undefined(self.offset_width) {
            return Err(Error::AppendUnsupported(
                "the dataset's extensible-array index block is not allocated",
            ));
        }
        let bytes = metadata::IndexBlockBytes::read(file, array)?;
        let index = bytes.parse()?;
        for reference in index.data_blocks() {
            metadata::require_range(file, reference?.range())?;
        }
        for reference in index.super_blocks() {
            metadata::check_block(file, reference?.framing())?;
        }
        let mut stats = ExtensibleArrayStats {
            nsuper_blks: 0,
            super_blk_size: 0,
            ndata_blks: 0,
            data_blk_size: 0,
            max_idx_set: self.num_chunks,
            nelmts: self.idx_blk_elmts,
        };
        let os = u64::from(self.offset_width.get());
        let pointers = ea_sum([
            self.index_block_addr.get(),
            EA_BLOCK_PREFIX_SIZE,
            os,
            ea_product(self.idx_blk_elmts, self.ea_elem_size.to_u64())?,
        ])?;
        for (ordinal, &elements) in self.geom.direct_dblk_nelmts().iter().enumerate() {
            let count = self.allocated_pointers(
                file,
                ea_sum([pointers, ea_product(ordinal.to_u64(), os)?])?,
                1,
                self.data_block_size(elements)?,
            )?;
            self.add_data_blocks(&mut stats, elements, count)?;
        }
        for (j, &(ndblks, elements)) in self
            .geom
            .sblks()
            .iter()
            .skip(self.geom.first_indirect_sblk())
            .take(self.geom.nsblk_addrs())
            .enumerate()
        {
            let addr = self.super_block_addr(file, j)?;
            if addr.is_undefined(self.offset_width) {
                continue;
            }
            let block = SuperBlockGeometry {
                ndblks,
                blocks: DataBlockGeometry {
                    dblk_nelmts: elements,
                    page_nelmts: self.ea_bits.page_nelmts().get(),
                },
            };
            self.add_super_block(&mut stats, block)?;
            let pointers = ea_sum([addr.get(), self.block_prefix_size(), block.bitmap_size()])?;
            let count =
                self.allocated_pointers(file, pointers, ndblks, self.data_block_size(elements)?)?;
            self.add_data_blocks(&mut stats, elements, count)?;
        }
        Ok(stats)
    }

    /// Counts defined addresses in `count` consecutive pointer slots at `offset`.
    ///
    /// Reads at most [`EA_POINTER_BATCH_BYTES`] bytes at a time.
    /// Resolves each defined address and checks the complete allocation of `allocation_length`
    /// bytes against the published EOF and physical source.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if a pointer offset exceeds
    /// [`u64::MAX`], [`Error::InvalidSpaceOwnershipExtent`] if an allocation ends past the
    /// published EOF, or [`FormatError::UnexpectedSourceEof`] if it ends past the physical source.
    /// Propagates address resolution, range overflow, and [`Source::read_at`] errors.
    fn allocated_pointers(
        &self,
        file: &impl Store,
        mut offset: u64,
        mut count: u64,
        allocation_length: u64,
    ) -> Result<u64, Error> {
        let os = usize::from(self.offset_width.get());
        let mut bytes = [0u8; EA_POINTER_BATCH_BYTES];
        let mut allocated = 0;
        while count > 0 {
            let take = count.min((bytes.len() / os).to_u64()).to_usize()?;
            let span = take * os;
            file.read_at(offset, &mut bytes[..span])?;
            for pointer in bytes[..span].chunks_exact(os) {
                let mut addr = [0u8; 8];
                addr[..os].copy_from_slice(pointer);
                let address = StoredAddress::new(u64::from_le_bytes(addr));
                if address.is_undefined(self.offset_width) {
                    continue;
                }
                metadata::require_range(
                    file,
                    FileRange::new(file.addresses().resolve(address)?, allocation_length)?,
                )?;
                allocated = ea_sum([allocated, 1])?;
            }
            offset = ea_sum([offset, span.to_u64()])?;
            count -= take.to_u64();
        }
        Ok(allocated)
    }

    /// Adds one secondary block's allocation count and encoded size to `stats`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if a size or statistic exceeds
    /// [`u64::MAX`].
    fn add_super_block(
        &self,
        stats: &mut ExtensibleArrayStats,
        block: SuperBlockGeometry,
    ) -> Result<(), FormatError> {
        let size = ea_sum([
            self.block_prefix_size(),
            ea_product(block.ndblks, block.blocks.npages().div_ceil(8))?,
            ea_product(block.ndblks, u64::from(self.offset_width.get()))?,
            EA_CHECKSUM_SIZE,
        ])?;
        stats.nsuper_blks = ea_sum([stats.nsuper_blks, 1])?;
        stats.super_blk_size = ea_sum([stats.super_blk_size, size])?;
        Ok(())
    }

    /// Adds `count` data blocks' allocation counts, encoded sizes, and element capacities.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if a size or statistic exceeds
    /// [`u64::MAX`].
    fn add_data_blocks(
        &self,
        stats: &mut ExtensibleArrayStats,
        elements: u64,
        count: u64,
    ) -> Result<(), FormatError> {
        if count == 0 {
            return Ok(());
        }
        stats.ndata_blks = ea_sum([stats.ndata_blks, count])?;
        stats.data_blk_size = ea_sum([
            stats.data_blk_size,
            ea_product(count, self.data_block_size(elements)?)?,
        ])?;
        stats.nelmts = ea_sum([stats.nelmts, ea_product(count, elements)?])?;
        Ok(())
    }

    /// Computes header statistics for a dense prefix of `num_chunks` element slots.
    ///
    /// Counts every data block in that prefix, including blocks whose pointers are undefined.
    /// Checks that the ranges, statistics, and metadata sizes are representable before an
    /// append reads or writes chunk data.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if an element range, block size,
    /// or statistic exceeds [`u64::MAX`], or [`FormatError::ValueTooLargeForPlatform`] if a block
    /// size or data-block index exceeds [`usize::MAX`]. Returns [`Error::AppendUnsupported`] if
    /// the prefix extends beyond the geometry.
    fn dense_prefix_stats(&self, num_chunks: u64) -> Result<ExtensibleArrayStats, Error> {
        self.planning_geometry().dense_prefix_stats(num_chunks)
    }

    /// Returns the encoded data-block size, including page storage and checksums.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the size exceeds [`u64::MAX`].
    fn data_block_size(&self, elements: u64) -> Result<u64, FormatError> {
        self.planning_geometry().data_block_size(elements)
    }

    /// Returns the prefix length through the block-offset field of a data or secondary block.
    ///
    /// Version 0 blocks are defined in "The Extensible Array Index" of the [format specification,
    /// version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
    fn block_prefix_size(&self) -> u64 {
        self.planning_geometry().block_prefix_size()
    }

    fn planning_geometry(&self) -> IndexPlanningGeometry<'_> {
        IndexPlanningGeometry {
            geom: &self.geom,
            idx_blk_elmts: self.idx_blk_elmts,
            ea_elem_size: self.ea_elem_size,
            ea_bits: self.ea_bits,
            offset_width: self.offset_width,
        }
    }
}

/// Block geometry and element encoding used to size an append before allocating storage.
struct IndexPlanningGeometry<'a> {
    geom: &'a ExtensibleArrayGeometry,
    idx_blk_elmts: u64,
    ea_elem_size: usize,
    ea_bits: ExtensibleArrayBitGeometry,
    offset_width: OffsetWidth,
}

impl IndexPlanningGeometry<'_> {
    /// Computes header statistics for a dense prefix, including complete block allocations.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if a range, size, or statistic
    /// exceeds [`u64::MAX`], or [`FormatError::ValueTooLargeForPlatform`] if a block size or
    /// index exceeds [`usize::MAX`]. Returns [`Error::AppendUnsupported`] if the prefix extends
    /// beyond the geometry.
    fn dense_prefix_stats(&self, num_chunks: u64) -> Result<ExtensibleArrayStats, Error> {
        if num_chunks > self.idx_blk_elmts {
            locate_data_block(self.geom, self.idx_blk_elmts, num_chunks - 1)?;
        }
        let mut stats = ExtensibleArrayStats {
            nsuper_blks: 0,
            super_blk_size: 0,
            ndata_blks: 0,
            data_blk_size: 0,
            max_idx_set: num_chunks,
            nelmts: self.idx_blk_elmts,
        };
        let mut elem = self.idx_blk_elmts;
        for &elements in self.geom.direct_dblk_nelmts() {
            if elem >= num_chunks {
                return Ok(stats);
            }
            let block_size = self.data_block_size(elements)?;
            block_size.to_usize()?;
            stats.ndata_blks = ea_sum([stats.ndata_blks, 1])?;
            stats.data_blk_size = ea_sum([stats.data_blk_size, block_size])?;
            stats.nelmts = ea_sum([stats.nelmts, elements])?;
            elem = ea_sum([elem, elements])?;
        }
        for &(ndblks, elements) in self
            .geom
            .sblks()
            .iter()
            .skip(self.geom.first_indirect_sblk())
            .take(self.geom.nsblk_addrs())
        {
            if elem >= num_chunks {
                return Ok(stats);
            }
            let span = ea_product(ndblks, elements)?;
            let end = ea_sum([elem, span])?;
            let used = (num_chunks.min(end) - elem).div_ceil(elements);
            let blocks = DataBlockGeometry {
                dblk_nelmts: elements,
                page_nelmts: self.ea_bits.page_nelmts().get(),
            };
            let bitmap = ea_product(ndblks, blocks.npages().div_ceil(8))?;
            let super_size = ea_sum([
                self.block_prefix_size(),
                bitmap,
                ea_product(ndblks, u64::from(self.offset_width.get()))?,
                EA_CHECKSUM_SIZE,
            ])?;
            super_size.to_usize()?;
            stats.nsuper_blks = ea_sum([stats.nsuper_blks, 1])?;
            stats.super_blk_size = ea_sum([stats.super_blk_size, super_size])?;
            stats.ndata_blks = ea_sum([stats.ndata_blks, used])?;
            let block_size = self.data_block_size(elements)?;
            block_size.to_usize()?;
            stats.data_blk_size = ea_sum([stats.data_blk_size, ea_product(used, block_size)?])?;
            stats.nelmts = ea_sum([stats.nelmts, ea_product(used, elements)?])?;
            elem = end;
        }
        Ok(stats)
    }

    fn data_block_size(&self, elements: u64) -> Result<u64, FormatError> {
        let blocks = DataBlockGeometry {
            dblk_nelmts: elements,
            page_nelmts: self.ea_bits.page_nelmts().get(),
        };
        if blocks.is_paged() {
            ea_sum([
                self.block_prefix_size(),
                EA_CHECKSUM_SIZE,
                ea_product(
                    blocks.npages(),
                    ea_sum([
                        ea_product(blocks.page_nelmts, self.ea_elem_size.to_u64())?,
                        EA_CHECKSUM_SIZE,
                    ])?,
                )?,
            ])
        } else {
            ea_sum([
                self.block_prefix_size(),
                ea_product(elements, self.ea_elem_size.to_u64())?,
                EA_CHECKSUM_SIZE,
            ])
        }
    }

    fn block_prefix_size(&self) -> u64 {
        EA_BLOCK_PREFIX_SIZE
            + u64::from(self.offset_width.get())
            + self.ea_bits.block_offset_width().bytes().to_u64()
    }
}

/// The result of planning one in-place Extensible-Array append, computed without a write: the
/// chunks to write and the counts the write phases publish.
pub(crate) struct AppendPlan {
    fields: AppendFields,
    /// The chunks to write, each with its stored bytes and its filter mask, in element order from
    /// `n_full`. The first is the rewritten trailing chunk when the current length is not a
    /// multiple of the chunk length.
    chunks: Vec<PreparedChunk>,
    /// Filtered replacement for a completed raw tail, published after the new extent.
    completed_tail: Option<PreparedChunk>,
    /// Captured tail reference and allocation retained until replacement publication.
    predecessor: Option<CapturedChunkReference>,
    /// Element index the first new chunk occupies.
    n_full: u64,
    /// New axis-0 dimension after the append (the commit value).
    new_dim: u64,
    /// Number of chunk positions within `new_dim`, including unallocated ones.
    new_num_chunks: u64,
    /// The header statistics to publish after the index writes.
    stats: ExtensibleArrayStats,
}

/// Encoded dimension and statistics fields paired with the widths used during planning.
struct AppendFields {
    widths: FormatWidths,
    dimension: [u8; 8],
    statistics: [u8; ExtensibleArrayStats::MAX_ENCODED_FIELDS_LEN],
    statistics_len: usize,
}

impl AppendFields {
    /// Checks and encodes the dimension and statistics into fixed-size buffers.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::LengthTooLarge`] if any value exceeds the selected length width.
    fn new(
        widths: FormatWidths,
        dimension: u64,
        stats: ExtensibleArrayStats,
    ) -> Result<Self, FormatError> {
        let mut fields = Self {
            widths,
            dimension: [0; 8],
            statistics: [0; ExtensibleArrayStats::MAX_ENCODED_FIELDS_LEN],
            statistics_len: 0,
        };
        hdf5_pure_format::__private::write_length_at(
            &mut fields.dimension,
            0,
            dimension,
            widths.lengths,
        )?;
        fields.statistics_len = stats.encode_fields(&mut fields.statistics, widths.lengths)?;
        Ok(fields)
    }

    /// Checks that publication uses the same widths as field planning.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if either width differs.
    fn require_widths(&self, widths: FormatWidths) -> Result<(), Error> {
        if self.widths != widths {
            return Err(Error::AppendUnsupported(
                "append field widths changed after planning",
            ));
        }
        Ok(())
    }
}

/// Plans chunk bytes, index counts, and header statistics for an in-place append.
///
/// Checks the element ranges and metadata sizes before reading a trailing chunk or encoding
/// `raw`, the little-endian bytes of `new_elems` elements. Computes header statistics from
/// referenced blocks and the blocks needed for the appended chunks. Planning leaves the file
/// unchanged. Encodes the dimension and statistics at the file's length width before reading
/// a trailing chunk or preparing new payloads.
///
/// Verifies existing data blocks and initialized pages touched by the append before allocating
/// storage. Metadata checksum verification is independent of the `checksum` feature.
///
/// `grow_visible_tail` permits rewriting a trailing chunk under a filter pipeline. An
/// exclusively locked session passes `true`, and the SWMR writer passes `false`. A partial
/// tail stored under lossy filters is rejected unless `lossy_tail_reencoding` permits rewriting
/// it. Under a lossy pipeline, a tail exempt from filtering remains raw until complete, then its
/// filtered replacement is published after the new extent.
///
/// # Errors
///
/// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if an index range, metadata size, or
/// header statistic exceeds [`u64::MAX`], or [`FormatError::ValueTooLargeForPlatform`] if metadata
/// sizes or indexes exceed [`usize::MAX`]. Returns [`FormatError::LengthTooLarge`] if the grown
/// dimension or a statistic exceeds the file's length width.
///
/// Returns [`Error::AppendUnsupported`] if the new dimension overflows, the index geometry does
/// not cover the append, a filtered tail rewrite is unsupported, or an encoded chunk size does
/// not fit its index record. Errors from reading the trailing chunk, constructing its datatype
/// context, verifying index metadata, splitting the element bytes, or encoding the chunks are
/// propagated.
pub(crate) fn plan_ea_append<F: Store>(
    file: &F,
    loc: &Located,
    datatype: &Datatype,
    spatial: &[u64],
    element_size: NonZeroUsize,
    pipeline: Option<&FilterPipeline>,
    grow_visible_tail: bool,
    lossy_tail_reencoding: LossyTailReencoding,
    raw: &[u8],
    new_elems: u64,
    fill: FillPattern<'_>,
) -> Result<AppendPlan, Error> {
    let chunk_elems = loc.chunk_elems;
    let current_dim = loc.current_dim;
    let new_dim = current_dim
        .checked_add(new_elems)
        .ok_or(Error::AppendUnsupported(
            "append would overflow the dataset dimension",
        ))?;
    if loc.offset_width != file.widths().offsets || loc.length_width != file.widths().lengths {
        return Err(Error::AppendUnsupported(
            "append file widths differ from the located dataset",
        ));
    }
    let mut encoded_dimension = [0; 8];
    hdf5_pure_format::__private::write_length_at(
        &mut encoded_dimension,
        0,
        new_dim,
        file.widths().lengths,
    )?;
    let new_num_chunks = new_dim.div_ceil(chunk_elems);
    loc.dense_prefix_stats(new_num_chunks)?;
    let n_full = current_dim / chunk_elems;
    loc.check_append_metadata(file, n_full, new_num_chunks)?;
    let stats = loc.planned_allocation_stats(file, n_full, new_num_chunks)?;
    let fields = AppendFields::new(file.widths(), new_dim, stats)?;
    let has_partial = current_dim % chunk_elems != 0;

    // `ChunkEncoding::decodable_by` selects bytes readable at both the old and new extents.
    // The index record and its checksum are published together by `Store::publish_checksummed`.
    // Lossless rewrites preserve the live prefix. With permission to re-encode a lossy tail,
    // the decoded prefix can change when the index record is published, before the new extent.
    //
    // When a raw tail becomes complete under a lossy pipeline, prepare a filtered copy as well.
    // Publish its record after the extent barrier: a reader at the old partial extent reads
    // the chunk's bytes verbatim, regardless of its filter mask. Before that barrier, the
    // raw copy preserves the old prefix. After it, either copy is readable at the new extent.
    if has_partial && let Some(pl) = pipeline {
        if !grow_visible_tail {
            return Err(Error::AppendUnsupported(
                "a filtered dataset whose length is not a whole multiple of the chunk length \
                 cannot be appended in place by this writer: growing its trailing partial chunk \
                 would repoint an index element a concurrent reader can already see. Use \
                 Dataset::append_staged, or append whole chunks",
            ));
        }
        if !loc.flags.partial_edge_chunks_stored_raw()
            && !edit::pipeline_lossless(pl)
            && lossy_tail_reencoding == LossyTailReencoding::Reject
        {
            return Err(Error::AppendUnsupported(LOSSY_TAIL_REFUSAL));
        }
    }

    // Build the raw tail region: the live prefix of any rewritten partial chunk,
    // then the appended bytes.
    let ctx = ChunkContext::from_datatype(spatial, datatype)?;
    // One encoder and decoder across the appended tail. See `FilterScratch`.
    let mut scratch = FilterScratch::new();
    let (mut tail_raw, predecessor) = if has_partial {
        let (bytes, reference) = loc.read_live_tail(file, &mut scratch, pipeline, ctx)?;
        (bytes, Some(reference))
    } else {
        (Vec::new(), None)
    };
    tail_raw.extend_from_slice(raw);

    // Split the tail into full chunk buffers and encode each for a reader at the current
    // dimension and one at the new dimension. The final chunk's slots past the new dimension
    // take the dataset's fill value (issue #296).
    let filter_completed_tail = has_partial
        && loc.flags.partial_edge_chunks_stored_raw()
        && new_dim / chunk_elems > n_full
        && pipeline.is_some_and(|pipeline| !edit::pipeline_lossless(pipeline));
    let (chunks, completed_tail) = prepare_append_chunks(
        &mut scratch,
        pipeline,
        ctx,
        spatial,
        element_size,
        loc.flags,
        chunk_elems,
        current_dim,
        new_dim,
        n_full,
        &tail_raw,
        fill,
        filter_completed_tail,
    )?;

    // Every blob has to fit the element it will be recorded in. Checked here,
    // while nothing has been written, rather than where the element is laid out
    // in phase 2 — by then phase 1 has already placed the chunk bytes, and the
    // refusal would leave them behind unreferenced.
    let os = file.widths().offsets.get() as usize;
    for chunk in chunks.iter().chain(completed_tail.iter()) {
        loc.check_stored_size(os, chunk.stored_size())?;
    }

    Ok(AppendPlan {
        fields,
        chunks,
        completed_tail,
        predecessor,
        n_full,
        new_dim,
        new_num_chunks,
        stats,
    })
}

/// Encodes padded chunks readable at both the current and appended extents.
///
/// If `filter_completed_tail` is set, also prepares a filtered replacement for the first chunk.
/// The caller publishes that replacement after the appended extent.
///
/// # Errors
///
/// Propagates chunk splitting, fill, and filter encoding errors.
fn prepare_append_chunks(
    scratch: &mut FilterScratch,
    pipeline: Option<&FilterPipeline>,
    ctx: ChunkContext<'_>,
    spatial: &[u64],
    element_size: NonZeroUsize,
    flags: ChunkedLayoutFlags,
    chunk_elems: u64,
    current_dim: u64,
    new_dim: u64,
    n_full: u64,
    tail_raw: &[u8],
    fill: FillPattern<'_>,
    filter_completed_tail: bool,
) -> Result<(Vec<PreparedChunk>, Option<PreparedChunk>), Error> {
    let tail_len_elems = new_dim - n_full * chunk_elems;
    let split =
        chunked_write::split_into_chunks(tail_raw, &[tail_len_elems], spatial, element_size, fill)
            .map_err(Error::Format)?;
    let mut completed_tail = None;
    let chunks = (n_full..)
        .zip(split)
        .map(|(e, logical)| {
            let offset = e * chunk_elems;
            if e == n_full && filter_completed_tail {
                completed_tail = Some(PreparedChunk::encode(
                    scratch,
                    pipeline,
                    ctx,
                    ChunkEncoding::Filtered,
                    logical.clone(),
                )?);
            }
            let current = (offset < current_dim).then(|| {
                ChunkDecoding::at_extent(flags, &[chunk_elems], &[current_dim], &[offset])
            });
            let encoding = ChunkEncoding::decodable_by(
                current,
                ChunkDecoding::at_extent(flags, &[chunk_elems], &[new_dim], &[offset]),
            );
            PreparedChunk::encode(scratch, pipeline, ctx, encoding, logical)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((chunks, completed_tail))
}

/// Prepares the first append and empty index geometry without allocating or writing storage.
///
/// Checks complete block sizes and encodes the payloads before the caller initializes the index.
///
/// # Errors
///
/// Returns [`Error::AppendUnsupported`] if the append exceeds the index geometry or an encoded
/// chunk exceeds its index record. Returns [`FormatError::LengthTooLarge`] if the dimension or
/// statistics exceed the file's length width. Propagates geometry, datatype, fill, and filter
/// encoding errors.
pub(crate) fn plan_initial_ea_append(
    index: &UnallocatedIndex,
    widths: FormatWidths,
    datatype: &Datatype,
    spatial: &[u64],
    element_size: NonZeroUsize,
    pipeline: Option<&FilterPipeline>,
    raw: &[u8],
    new_elems: u64,
    fill: FillPattern<'_>,
) -> Result<(AppendPlan, ExtensibleArrayInitialization), Error> {
    if widths != index.widths {
        return Err(Error::AppendUnsupported(
            "append file widths differ from the located dataset",
        ));
    }
    hdf5_pure_format::__private::write_length_at(&mut [0; 8], 0, new_elems, widths.lengths)?;
    let initialization = ExtensibleArrayInitialization::new(
        index.creation,
        index.chunk_bytes.to_u64(),
        widths,
        pipeline.is_some(),
    )?;
    let geometry = initialization.geometry()?;
    let encoding = initialization.element_encoding();
    let num_chunks = new_elems.div_ceil(index.chunk_elems);
    let stats = IndexPlanningGeometry {
        geom: &geometry,
        idx_blk_elmts: u64::from(index.creation.inline_elements()),
        ea_elem_size: encoding.elem_size,
        ea_bits: index.creation.bit_geometry(),
        offset_width: widths.offsets,
    }
    .dense_prefix_stats(num_chunks)?;
    let fields = AppendFields::new(widths, new_elems, stats)?;
    let ctx = ChunkContext::from_datatype(spatial, datatype)?;
    let (chunks, completed_tail) = prepare_append_chunks(
        &mut FilterScratch::new(),
        pipeline,
        ctx,
        spatial,
        element_size,
        index.flags,
        index.chunk_elems,
        0,
        new_elems,
        0,
        raw,
        fill,
        false,
    )?;
    if pipeline.is_some()
        && chunks
            .iter()
            .any(|chunk| !encoding.holds_stored_size(chunk.stored_size()))
    {
        return Err(Error::AppendUnsupported(
            "encoded chunk size exceeds the dataset's extensible-array element width",
        ));
    }
    Ok((
        AppendPlan {
            fields,
            chunks,
            completed_tail,
            predecessor: None,
            n_full: 0,
            new_dim: new_elems,
            new_num_chunks: num_chunks,
            stats,
        },
        initialization,
    ))
}

/// Allocates and publishes an empty Extensible Array before returning its located geometry.
///
/// Checks the initialization plan's widths and creation parameters against the store and layout.
/// Encodes the index for its reserved address, confirms allocation ownership, publishes the
/// layout's index-header reference, and synchronizes it before rereading the index. Marks
/// allocation publication uncertain if that final read fails.
///
/// # Errors
///
/// Returns [`Error::AppendUnsupported`] if initialization properties differ from the file or
/// layout, or [`FormatError::Internal`] if the published index remains unallocated. Propagates
/// allocation, encoding, ownership, reference-publication, synchronization, and locate errors.
pub(crate) fn initialize_ea_index(
    file: &mut impl Store,
    oh_addr: ObjectId,
    index: &UnallocatedIndex,
    initialization: &ExtensibleArrayInitialization,
) -> Result<Located, Error> {
    if initialization.plan().widths() != file.widths()
        || index.widths != file.widths()
        || initialization.plan().creation() != index.creation
    {
        return Err(Error::AppendUnsupported(
            "index initialization differs from the layout or file widths",
        ));
    }
    let (header, index_allocation) = file.allocate_ea_initialization(initialization)?;
    let publication = file.publish_allocations()?;
    if let Some(index_allocation) = index_allocation {
        publication.confirm(index_allocation)?;
    }
    let allocation = publication.confirm(header)?;
    index.reference.publish(file, &allocation)?;
    file.sync()?;
    let result =
        Located::locate_at(file, oh_addr, Error::AppendUnsupported).and_then(
            |located| match located.located {
                AppendIndex::Allocated(index) => Ok(index),
                AppendIndex::Fixed(_) | AppendIndex::Single(_) | AppendIndex::Unallocated(_) => {
                    Err(FormatError::Internal("published index remains unallocated".into()).into())
                }
            },
        );
    if result.is_err() {
        file.make_allocation_uncertain();
    }
    result
}

/// Publishes a planned append with synchronization barriers between its storage updates.
///
/// Writes each child before publishing its address, with synchronization barriers between
/// phases. Runs phase 1, then continues through `max_phase`, up to phase 4. Production callers
/// pass 4. Tests stop at a phase boundary to simulate interruption.
///
/// Caches allocation statistics after phase 3's header publication barrier. Updates
/// [`Located::current_dim`] and [`Located::num_chunks`] after phase 4's dimension publication
/// barrier. If a raw tail becomes complete under a lossy pipeline, publishes its filtered
/// replacement after that barrier. An interruption can leave the completed chunk unfiltered.
///
/// Transfers predecessor ownership to [`Store::retire`] after confirmed index publication. The
/// original tail and any intermediate raw completion have separate publication barriers. The
/// store retains confirmed retirements if a later phase fails.
///
/// # Errors
///
/// Returns [`Error::Format`] if a file offset cannot be represented or encoded block metadata
/// is invalid. Returns [`Error::AppendUnsupported`] if an index element or publication range
/// lies outside its block, or a captured tail reference belongs to another index or has changed
/// since planning. Propagates errors from allocating, reading, writing, or synchronizing file
/// storage. Returns [`Error::AppendUnsupported`] before allocation if the file's widths differ
/// from those used to encode the planned fields.
pub(crate) fn apply_ea_append<F: Store>(
    file: &mut F,
    loc: &mut Located,
    AppendPlan {
        fields,
        chunks,
        completed_tail,
        predecessor,
        n_full,
        new_dim,
        new_num_chunks,
        stats,
    }: AppendPlan,
    max_phase: u8,
) -> Result<(), Error> {
    fields.require_widths(file.widths())?;
    // Phase 1: write payloads and confirm their publication before index updates. A replacement
    // tail is visible at the old dimension, so its complete allocation must be within recorded EOF.
    let mut allocations = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        allocations.push(file.allocate(chunk.bytes(), AllocationClass::Raw)?);
    }
    let completed_allocation = completed_tail
        .as_ref()
        .map(|chunk| file.allocate(chunk.bytes(), AllocationClass::Raw))
        .transpose()?;
    let publication = file.publish_allocations()?;
    let completed_record = completed_allocation
        .zip(completed_tail.as_ref())
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

    // Phase 2: insert new chunk records and repoint the trailing chunk's record.
    // Each new child block is published before its parent pointer is written.
    let mut records = (n_full..).zip(allocations.into_iter().zip(&chunks).map(
        |(allocation, chunk)| {
            publication
                .confirm(allocation)
                .map(|allocation| ConfirmedChunk {
                    allocation,
                    filter_mask: chunk.filter_mask(),
                })
        },
    ));
    if let Some(reference) = predecessor {
        let (index, record) = records
            .next()
            .ok_or_else(|| FormatError::Internal("a tail replacement has no record".into()))?;
        debug_assert_eq!(reference.slot, index);
        UnpublishedChunkReference::bind(file, loc, reference)?
            .publish(record?, records.map(|(_, record)| record))?;
    } else {
        for (index, record) in records {
            loc.ea_insert(file, index, record?)?;
        }
        file.sync()?;
    }
    if max_phase < 3 {
        return Ok(());
    }

    // Phase 3: publish the EA header element count.
    loc.publish_statistics(file, &fields.statistics[..fields.statistics_len])?;
    file.sync()?;
    loc.allocation_stats = Some(stats);
    if max_phase < 4 {
        return Ok(());
    }

    // Phase 4: publish the dataspace dimension before publishing a filtered tail.
    loc.dimension.publish(
        file,
        &fields.dimension[..usize::from(fields.widths.lengths.get())],
    )?;
    file.sync()?;

    loc.current_dim = new_dim;
    loc.num_chunks = new_num_chunks;
    if let Some(record) = completed_record {
        UnpublishedChunkReference::read(file, loc, n_full)?.publish(record, [])?;
        loc.allocation_stats = Some(stats);
    }
    Ok(())
}

/// A confirmed payload allocation and its encoded filter mask.
struct ConfirmedChunk {
    allocation: ConfirmedAllocation,
    filter_mask: u32,
}

impl ConfirmedChunk {
    /// Resolves the allocation through its editing session before constructing an index record.
    ///
    /// # Errors
    ///
    /// Returns the allocation-state or stored-address conversion error.
    fn record(&self, file: &impl Store) -> Result<ElemRecord, Error> {
        let range = file.allocation_publication().require(&self.allocation)?;
        Ok(ElemRecord {
            addr: file.addresses().relative(range.start())?,
            stored_size: range.length(),
            filter_mask: self.filter_mask,
        })
    }
}

/// A predecessor allocation and index record captured at the current dataset extent.
struct CapturedChunkReference {
    owner: ChunkReferenceOwner,
    slot: u64,
    dimension: u64,
    record: ElemRecord,
    allocation: ChunkAllocation,
}

/// Index, dataspace-field, and base-address coordinates used to bind a captured reference.
#[derive(Eq, PartialEq)]
struct ChunkReferenceOwner {
    index_header: StoredAddress,
    dimension_offset: u64,
    base: BaseAddress,
}

impl ChunkReferenceOwner {
    fn new(file: &impl Store, located: &Located) -> Self {
        Self {
            index_header: located.ea_addr,
            dimension_offset: located.dimension.field.start().get(),
            base: file.addresses().effective_base(),
        }
    }
}

/// A checked predecessor held with exclusive access for one replacement publication.
struct UnpublishedChunkReference<'a, F: Store> {
    file: &'a mut F,
    located: &'a mut Located,
    slot: u64,
    allocation: ChunkAllocation,
}

impl<'a, F: Store> UnpublishedChunkReference<'a, F> {
    /// Binds a captured predecessor after comparing its owner coordinates, dimension, and record.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if the captured reference belongs to another index, or
    /// its dimension or record has changed. Propagates record-read errors.
    fn bind(
        file: &'a mut F,
        located: &'a mut Located,
        captured: CapturedChunkReference,
    ) -> Result<Self, Error> {
        let CapturedChunkReference {
            owner,
            slot,
            dimension,
            record,
            allocation,
        } = captured;
        if owner != ChunkReferenceOwner::new(file, located) {
            return Err(Error::AppendUnsupported(
                "append plan refers to a different chunk index",
            ));
        }
        if dimension != located.current_dim || located.read_element(file, slot)? != Some(record) {
            return Err(Error::AppendUnsupported(
                "chunk reference changed after append planning",
            ));
        }
        Ok(Self {
            file,
            located,
            slot,
            allocation,
        })
    }

    /// Reads the current index record and captures ownership of its allocation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if the slot is unallocated. Propagates metadata and
    /// allocation-geometry errors.
    fn read(file: &'a mut F, located: &'a mut Located, slot: u64) -> Result<Self, Error> {
        let record = located
            .read_element(file, slot)?
            .ok_or(Error::AppendUnsupported(
                "replaced chunk is missing from the index",
            ))?;
        let allocation = ChunkAllocation::capture(file.addresses(), &located.chunk_info(record))?;
        Ok(Self {
            file,
            located,
            slot,
            allocation,
        })
    }

    /// Writes replacement and following records, then confirms retirement after the ordering
    /// barrier.
    ///
    /// On a write or barrier error, the predecessor and replacement allocations remain unavailable
    /// for reuse.
    ///
    /// # Errors
    ///
    /// Returns the index-write or synchronization error, or [`Error::Format`] if a following
    /// record's index exceeds [`u64::MAX`].
    fn publish(
        self,
        record: ConfirmedChunk,
        following: impl IntoIterator<Item = Result<ConfirmedChunk, Error>>,
    ) -> Result<(), Error> {
        let Self {
            file,
            located,
            slot,
            allocation,
        } = self;
        let pending = UncertainChunkReference {
            file,
            located,
            slot,
            allocation,
        };
        pending
            .located
            .ea_insert(pending.file, pending.slot, record)?;
        let mut next = pending.slot;
        for record in following {
            next = next
                .checked_add(1)
                .ok_or_else(|| FormatError::Internal("append record index exceeds u64".into()))?;
            pending.located.ea_insert(pending.file, next, record?)?;
        }
        pending.confirm()
    }
}

/// A predecessor held while replacement publication is unconfirmed.
struct UncertainChunkReference<'a, F: Store> {
    file: &'a mut F,
    located: &'a mut Located,
    slot: u64,
    allocation: ChunkAllocation,
}

impl<F: Store> UncertainChunkReference<'_, F> {
    /// Completes the ordering barrier and transfers the predecessor to [`Store::retire`].
    ///
    /// # Errors
    ///
    /// Returns the synchronization error before transferring ownership if the barrier fails.
    fn confirm(self) -> Result<(), Error> {
        self.file.sync()?;
        self.file.retire(RetiredChunk {
            allocation: self.allocation,
        });
        Ok(())
    }
}

/// A predecessor allocation whose replacement passed the index publication barrier.
pub(crate) struct RetiredChunk {
    allocation: ChunkAllocation,
}

impl RetiredChunk {
    #[cfg(test)]
    fn range(&self) -> FileRange {
        self.allocation.range()
    }

    /// Consumes a confirmed retirement into its stored-address extent and free-space class.
    pub(crate) fn into_extent(self) -> (Extent, FreeClass) {
        self.allocation.into_extent()
    }
}

/// The byte of a super block's page-init bitmap that holds page `global_page`'s
/// bit (MSB-first), and that byte with the bit set.
///
/// A read-modify-write, so it reads the byte and hands the *result* back for the
/// publish to place: one field of one byte is not a value the caller can name on
/// its own.
fn sb_page_bit<F: Store>(
    file: &F,
    sblk_addr: StoredAddress,
    blk_off: BlockOffsetWidth,
    global_page: u64,
) -> Result<(u64, u8), Error> {
    let os = file.widths().offsets.get() as usize;
    let bitmap_start = sblk_addr.get() + (4 + 1 + 1 + os + blk_off.bytes()) as u64;
    let byte = bitmap_start + global_page / 8;
    let mut v = [0u8; 1];
    file.read_at(byte, &mut v)?;
    Ok((byte, v[0] | (0x80u8 >> (global_page % 8))))
}

/// File offset of the `dblk_local`-th data-block-address slot inside a super
/// block, accounting for the page-init bitmap when the block is paged.
fn sb_dblk_slot_off(
    os: usize,
    sblk_addr: StoredAddress,
    dblk_local: usize,
    sb: SuperBlockGeometry,
    blk_off: BlockOffsetWidth,
) -> u64 {
    let prefix = (4 + 1 + 1 + os + blk_off.bytes()) as u64;
    sblk_addr.get() + prefix + sb.bitmap_size() + (dblk_local * os) as u64
}

// ---------------------------------------------------------------------------
// Extensible-array element location (where element `e` lives in the structure)
// ---------------------------------------------------------------------------

enum Parent {
    /// Reached by a direct data-block pointer in the index block.
    IndexDirect { ordinal: usize },
    /// Reached via a super block (`sblk_j`-th super-block pointer), as the
    /// `dblk_local`-th data block within it.
    Super { sblk_j: usize, dblk_local: usize },
}

struct DataBlockLoc {
    db_start: u64,
    dblk_nelmts: u64,
    ndblks: u64,
    sb_block_offset: u64,
    parent: Parent,
}

impl DataBlockLoc {
    /// The shape of the super block this data block belongs to, in an array
    /// whose pages hold `page_nelmts` elements.
    ///
    /// Meaningful for a [`Parent::Super`] block; for a [`Parent::IndexDirect`]
    /// one the counts describe the single block itself and no `EASB` exists to
    /// apply them to.
    fn super_block(&self, page_nelmts: u64) -> SuperBlockGeometry {
        SuperBlockGeometry {
            ndblks: self.ndblks,
            blocks: self.blocks(page_nelmts),
        }
    }

    /// How this data block is paged, which holds whether or not it has a super
    /// block above it.
    fn blocks(&self, page_nelmts: u64) -> DataBlockGeometry {
        DataBlockGeometry {
            dblk_nelmts: self.dblk_nelmts,
            page_nelmts,
        }
    }
}

/// Locates a non-inline element in the direct or secondary data-block ranges.
///
/// # Errors
///
/// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if `e` is an inline element or a
/// range calculation exceeds [`u64::MAX`], or [`FormatError::ValueTooLargeForPlatform`] if the
/// data-block index exceeds [`usize::MAX`]. Returns [`Error::AppendUnsupported`] if the geometry
/// does not cover `e`.
fn locate_data_block(
    geom: &ExtensibleArrayGeometry,
    idx_blk_elmts: u64,
    e: u64,
) -> Result<DataBlockLoc, Error> {
    if e < idx_blk_elmts {
        return Err(FormatError::InvalidExtensibleArrayGeometry.into());
    }
    let mut elem = idx_blk_elmts;
    for (ordinal, &dn) in geom.direct_dblk_nelmts().iter().enumerate() {
        let end = ea_sum([elem, dn])?;
        if e < end {
            return Ok(DataBlockLoc {
                db_start: elem,
                dblk_nelmts: dn,
                ndblks: 1,
                sb_block_offset: 0,
                parent: Parent::IndexDirect { ordinal },
            });
        }
        elem = end;
    }
    for (j, &(ndblks, dn)) in geom
        .sblks()
        .iter()
        .skip(geom.first_indirect_sblk())
        .take(geom.nsblk_addrs())
        .enumerate()
    {
        let span = ea_product(ndblks, dn)?;
        let end = ea_sum([elem, span])?;
        if e < end {
            let sb_block_offset = elem - idx_blk_elmts;
            let within = e - elem;
            let dblk_local = (within / dn).to_usize()?;
            let db_start = ea_sum([elem, ea_product(dblk_local.to_u64(), dn)?])?;
            return Ok(DataBlockLoc {
                db_start,
                dblk_nelmts: dn,
                ndblks,
                sb_block_offset,
                parent: Parent::Super {
                    sblk_j: j,
                    dblk_local,
                },
            });
        }
        elem = end;
    }
    Err(Error::AppendUnsupported(
        "chunk index geometry does not cover the appended element",
    ))
}

fn ea_sum(values: impl IntoIterator<Item = u64>) -> Result<u64, FormatError> {
    values.into_iter().try_fold(0u64, |total, value| {
        total
            .checked_add(value)
            .ok_or(FormatError::InvalidExtensibleArrayGeometry)
    })
}

fn ea_product(left: u64, right: u64) -> Result<u64, FormatError> {
    left.checked_mul(right)
        .ok_or(FormatError::InvalidExtensibleArrayGeometry)
}

// ---------------------------------------------------------------------------
// Object-header walk that records message file offsets + chunk checksum regions
// ---------------------------------------------------------------------------

struct WalkedMessage {
    msg_type: MessageType,
    /// The record's flags. With [`MessageFlags::SHARED`] set, the body is a reference
    /// to a shared message, which every parse below would otherwise decode as content.
    flags: MessageFlags,
    /// Absolute file offset of the message body.
    data_off: u64,
    size: usize,
    /// Containing chunk's checksum coverage: `[chunk_start, chunk_msg_end)`, with
    /// the 4-byte checksum stored at `chunk_msg_end`.
    chunk_start: u64,
    chunk_msg_end: u64,
}

struct Walk {
    messages: Vec<WalkedMessage>,
}

/// Walks a version 2 object header, chunk 0 and every continuation block, and
/// records, for each message other than a continuation, the absolute offset of
/// its body and the checksum region of its chunk.
///
/// The walk works over a store with no whole-file mirror. It reads the first
/// [`OBJECT_HEADER_PREFIX_MAX_LEN`] bytes of the header, or fewer where the file
/// ends sooner, then the message region of chunk 0, then one window per
/// continuation block.
///
/// # Errors
///
/// Returns [`Error::Format`] if the header ends inside its prefix or does not
/// begin with its signature, if the version is not 2, if a continuation block
/// lacks its signature or is too short for it and its checksum, if a chunk runs
/// past the end of `source`, if the body of a message record runs past the end
/// of its chunk, if a continuation message ends inside its fields or stores the
/// undefined address, or if the header has more than 255 continuation blocks.
fn walk_v2_object_header<S: Source + ?Sized>(
    source: &S,
    offset: PhysicalAddress,
    widths: FormatWidths,
) -> Result<Walk, Error> {
    let offset = offset.get();
    let head_len = source
        .len()
        .saturating_sub(offset)
        .min(OBJECT_HEADER_PREFIX_MAX_LEN.to_u64())
        .to_usize()?;
    let head = source.read_metadata_at(offset, head_len)?;
    let prefix = ObjectHeaderPrefix::parse(&head)?;
    let chunk0_msg_start = offset + prefix.len.to_u64();
    let chunk0 = source.read_metadata_at(chunk0_msg_start, prefix.chunk0_size.to_usize()?)?;
    let layout = prefix.prefix.layout;
    let mut messages = Vec::new();
    let mut continuations: Vec<(u64, usize)> = Vec::new();
    walk_messages(
        &chunk0,
        chunk0_msg_start,
        offset,
        chunk0_msg_start + chunk0.len().to_u64(),
        layout,
        widths,
        &mut messages,
        &mut continuations,
    )?;

    let mut guard = 256;
    while let Some((cont_off, cont_len)) = continuations.pop() {
        guard -= 1;
        if guard == 0 {
            return Err(Error::Format(FormatError::NestingDepthExceeded));
        }
        let block = source.read_metadata_at(cont_off, cont_len)?;
        let messages_range = hdf5_pure_format::__private::continuation_block_messages(&block)
            .map_err(|err| {
                Error::Format(match err {
                    FormatError::UnexpectedEof { .. } => FormatError::InvalidObjectHeaderSignature,
                    err => err,
                })
            })?;
        let region = block.get(messages_range.clone()).ok_or_else(|| {
            FormatError::Internal("a continuation block's messages lie outside the block".into())
        })?;
        walk_messages(
            region,
            cont_off + messages_range.start.to_u64(),
            cont_off,
            cont_off + messages_range.end.to_u64(),
            layout,
            widths,
            &mut messages,
            &mut continuations,
        )?;
    }

    Ok(Walk { messages })
}

/// Appends each message of one object header chunk to `messages`, except a
/// continuation message, whose block location it appends to `continuations`.
///
/// `chunk` holds the message region of the chunk, whose records are in
/// `layout`, and `base` is the absolute file offset of `chunk[0]`, so recorded
/// message offsets are absolute.
///
/// # Errors
///
/// Returns [`Error::Format`] if the body of a record runs past the end of
/// `chunk`, or if a continuation message ends inside its fields or stores the
/// undefined address.
#[allow(clippy::too_many_arguments)]
fn walk_messages(
    chunk: &[u8],
    base: u64,
    chunk_start: u64,
    chunk_msg_end: u64,
    layout: MessageRecordLayout,
    widths: FormatWidths,
    messages: &mut Vec<WalkedMessage>,
    continuations: &mut Vec<(u64, usize)>,
) -> Result<(), Error> {
    let mut msg_start = 0;
    while let Some(record) = layout.next_message(chunk, msg_start)? {
        let MessageRecord {
            msg_type,
            flags,
            body,
            body_range,
            ..
        } = record;
        if msg_type == MessageType::OBJECT_HEADER_CONTINUATION {
            let continuation = ObjectHeaderContinuation::parse_with_widths(body, widths)?;
            continuations.push((
                continuation.address().get(),
                continuation.length().to_usize()?,
            ));
        } else {
            messages.push(WalkedMessage {
                msg_type,
                flags,
                data_off: base + body_range.start.to_u64(),
                size: body.len(),
                chunk_start,
                chunk_msg_end,
            });
        }
        msg_start = body_range.end;
    }
    Ok(())
}

/// The signature, version, and client-ID width of a version 0 data or secondary block.
///
/// Defined in "The Extensible Array Index" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
const EA_BLOCK_PREFIX_SIZE: u64 = 6;
/// The checksum width of a version 0 Extensible Array block or data-block page.
///
/// Defined in "The Extensible Array Index" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
const EA_CHECKSUM_SIZE: u64 = 4;
/// The maximum read size when counting allocated block pointers.
const EA_POINTER_BATCH_BYTES: usize = 1_024;

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::cell::RefCell;

    use hdf5_pure_core::__private::FileRange;
    use hdf5_pure_core::__private::FileRegion;
    use hdf5_pure_core::PhysicalAddress;
    use hdf5_pure_format::__private::LAYOUT_PROPERTIES_OFFSET;
    use hdf5_pure_format::__private::LayoutVersion;
    use rstest::rstest;
    use test_util::image::Image;
    use test_util::object_header::Message;
    use test_util::object_header::MessageType as RecordType;
    use test_util::object_header::v2 as v2_bytes;
    use test_util::widths::Widths;

    use super::metadata as metadata_io;
    use super::metadata::IndexBlockBytes;
    use super::*;
    use crate::DatasetCreateProperties;
    use crate::PartialEdgeFiltering;
    use crate::ScaleOffset;
    use crate::access_mode::AccessMode;
    use crate::edit::LocatedState;
    use crate::edit::allocation::PublicationAction;
    use crate::filters;
    use crate::group::v2 as group_v2;
    use crate::object_path::ObjectPath;
    use crate::source::BytesSource;
    use crate::superblock::LocatedSuperblock;
    use crate::superblock::Superblock;
    use crate::writer::FileBuilder;

    mod fixed;
    mod metadata;
    mod single;

    test_util_hdf5::hdf5_1_10_tests! {
        mod native_crosschecks;
    }

    #[rstest]
    #[case::two(OffsetWidth::Two)]
    #[case::four(OffsetWidth::Four)]
    #[case::eight(OffsetWidth::Eight)]
    fn version_one_dataspace_publication_preserves_its_reserved_prefix(
        #[case] offsets: OffsetWidth,
    ) {
        let mut builder = FileBuilder::new();
        builder.with_offset_width(offsets);
        builder
            .create_dataset("d")
            .with_i32_data(&[0, 1, 2])
            .with_shape(&[3])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4]);
        let mut store = WindowProbeStore::open(builder.finish().unwrap());
        let walk = walk_v2_object_header(
            &store,
            dataset_address(&store).header_address(),
            store.widths,
        )
        .unwrap();
        let mut header = v2_bytes::Header::new();
        for message in walk.messages {
            let body = if message.msg_type == MessageType::DATASPACE {
                test_util::dataspace::v1(
                    1,
                    test_util::dataspace::Flags::MAX_DIMS,
                    &[3],
                    Some(&[u64::MAX]),
                )
            } else {
                store.read_exact_at(message.data_off, message.size).unwrap()
            };
            header = header.message(Message::new(RecordType(message.msg_type.to_u16()), &body));
        }
        let address = store.append_bytes(&header.build()).unwrap();
        store.patch_superblock_eof().unwrap();
        let mut state =
            edit::locate_dataset_state(&store, ObjectId::new(PhysicalAddress::new(address)))
                .unwrap();
        let walk =
            walk_v2_object_header(&store, PhysicalAddress::new(address), store.widths).unwrap();
        let dataspace = walk
            .messages
            .iter()
            .find(|message| message.msg_type == MessageType::DATASPACE)
            .unwrap();
        let field = dataspace.data_off.to_usize().unwrap();
        let prefix = store.data[field..field + test_util::dataspace::V1_HEADER_LEN].to_vec();
        append_state(&mut store, &mut state, &[3, 4], 4).unwrap();
        assert_eq!(
            &store.data[field..field + test_util::dataspace::V1_HEADER_LEN],
            prefix
        );
        let dimension = field + test_util::dataspace::V1_HEADER_LEN;
        assert_eq!(&store.data[dimension..dimension + 8], 5_u64.to_le_bytes());
        assert_eq!(
            &store.data[dimension + 8..dimension + 16],
            u64::MAX.to_le_bytes()
        );
        assert_eq!(state.index.current_dim(), 5);
        let checksum = test_util::checksum::lookup3(
            &store.data[dataspace.chunk_start.to_usize().unwrap()
                ..dataspace.chunk_msg_end.to_usize().unwrap()],
        );
        assert_eq!(
            &store.data[dataspace.chunk_msg_end.to_usize().unwrap()
                ..(dataspace.chunk_msg_end + EA_CHECKSUM_SIZE)
                    .to_usize()
                    .unwrap()],
            checksum.to_le_bytes()
        );
    }

    #[rstest]
    #[case::relative_span(0, u64::MAX, EA_CHECKSUM_SIZE)]
    #[case::physical_end(8, 8, u64::MAX - EA_CHECKSUM_SIZE)]
    fn checksummed_publication_rejects_a_wrapped_physical_end_before_reading(
        #[case] start: u64,
        #[case] expected_offset: u64,
        #[case] expected_length: u64,
    ) {
        let mut store = WindowProbeStore::open(build_unlimited(3, 4));
        let original = store.data.clone();
        store.reset_counters();
        let error = store
            .publish_checksummed(start, u64::MAX, start, &[1])
            .unwrap_err();
        let Error::Format(FormatError::OffsetOverflow { offset, length }) = error else {
            panic!("expected OffsetOverflow, got {error:?}");
        };
        assert_eq!((offset, length), (expected_offset, expected_length));
        assert_eq!(store.data, original);
        assert_eq!(store.total_read.get(), 0);
        assert!(store.allocations.borrow().is_empty());
        assert_eq!(store.syncs.get(), 0);
    }

    #[rstest]
    #[case::recorded_end(true)]
    #[case::physical_end(false)]
    fn dimension_publication_checks_the_complete_header_before_reading(#[case] recorded: bool) {
        let mut store = WindowProbeStore::open(build_unlimited(3, 4));
        let (index, _) = locate(&store);
        let range = index.dimension.chunk;
        if recorded {
            store.superblock.eof_address = range.end().get() - 1;
        } else {
            store
                .data
                .truncate((range.end().get() - 1).to_usize().unwrap());
        }
        let original = store.data.clone();
        store.reset_counters();
        let error = index
            .dimension
            .publish(&mut store, &5_u64.to_le_bytes())
            .unwrap_err();
        if recorded {
            let Error::InvalidSpaceOwnershipExtent {
                address,
                length,
                previous_end,
                logical_size,
            } = error
            else {
                panic!("expected InvalidSpaceOwnershipExtent, got {error:?}");
            };
            assert_eq!(
                (address, length, previous_end, logical_size),
                (
                    range.start().get(),
                    range.length(),
                    0,
                    range.end().get() - 1
                )
            );
        } else {
            let Error::Format(FormatError::UnexpectedSourceEof {
                expected,
                available,
            }) = error
            else {
                panic!("expected UnexpectedSourceEof, got {error:?}");
            };
            assert_eq!(
                (expected, available),
                (range.end().get(), range.end().get() - 1)
            );
        }
        assert_eq!(store.data, original);
        assert_eq!(store.total_read.get(), 0);
        assert_eq!(store.syncs.get(), 0);
        assert!(store.allocations.borrow().is_empty());
    }

    #[rstest]
    #[case::offsets(FormatWidths { offsets: OffsetWidth::Two, lengths: LengthWidth::Eight })]
    #[case::lengths(FormatWidths { offsets: OffsetWidth::Eight, lengths: LengthWidth::Two })]
    fn append_publication_rejects_changed_widths_before_allocating(#[case] widths: FormatWidths) {
        let mut store = WindowProbeStore::open(build_unlimited(3, 4));
        let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
        let plan = state_plan(&store, &state, &[3, 4]).unwrap();
        let original = store.data.clone();
        store.reset_counters();
        store.widths = widths;
        let error =
            apply_ea_append(&mut store, state.index.allocated_mut().unwrap(), plan, 4).unwrap_err();
        assert!(
            matches!(
                error,
                Error::AppendUnsupported("append field widths changed after planning")
            ),
            "{error:?}"
        );
        assert_eq!(store.data, original);
        assert!(store.allocations.borrow().is_empty());
        assert!(store.retired.is_empty());
        assert_eq!(store.syncs.get(), 0);
        assert_eq!(state.index.current_dim(), 3);
    }

    #[rstest]
    #[case::two(LengthWidth::Two, u64::from(u16::MAX) + 1)]
    #[case::four(LengthWidth::Four, u64::from(u32::MAX) + 1)]
    fn append_dimension_overflow_is_rejected_before_metadata_or_payload_reads(
        #[case] lengths: LengthWidth,
        #[case] dimension: u64,
    ) {
        let mut store = WindowProbeStore::open(build_unlimited(3, 4));
        let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
        store.widths.lengths = lengths;
        let index = state.index.allocated_mut().unwrap();
        index.length_width = lengths;
        index.current_dim = dimension - 1;
        let original = store.data.clone();
        store.reset_counters();
        let Err(error) = plan_ea_append(
            &store,
            state.index.allocated().unwrap(),
            &state.datatype,
            &state.spatial,
            state.element_size,
            state.pipeline.as_ref(),
            true,
            LossyTailReencoding::Reject,
            &[],
            1,
            state.fill.pattern(state.element_size),
        ) else {
            panic!("a dimension past the length width must be rejected");
        };
        let Error::Format(FormatError::LengthTooLarge {
            length,
            length_size,
        }) = error
        else {
            panic!("expected LengthTooLarge, got {error:?}");
        };
        assert_eq!((length, length_size), (dimension, lengths.get()));
        assert_eq!(store.data, original);
        assert_eq!(store.total_read.get(), 0);
        assert!(store.allocations.borrow().is_empty());
        assert!(store.retired.is_empty());
    }

    #[rstest]
    #[case::sentinel(65_535)]
    #[case::one_beyond(65_536)]
    fn narrow_append_allocation_rejects_an_unrepresentable_end_before_writes(#[case] end: u64) {
        let mut builder = FileBuilder::new();
        builder.with_offset_width(OffsetWidth::Two);
        builder
            .create_dataset("d")
            .with_i32_data(&[0, 1, 2])
            .with_shape(&[3])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4]);
        let mut store = WindowProbeStore::open(builder.finish().unwrap());
        store.data.resize(65_530, 0);
        store.patch_superblock_eof().unwrap();
        let allocation = store.allocate(&[1; 4], AllocationClass::Raw).unwrap();
        assert_eq!(allocation.range().end().get(), 65_534);
        store
            .publish_allocations()
            .unwrap()
            .confirm(allocation)
            .unwrap();
        let original = store.data.clone();
        store.reset_counters();
        let Err(error) = store.allocate(
            &vec![2; (end - 65_534).to_usize().unwrap()],
            AllocationClass::Raw,
        ) else {
            panic!("an unrepresentable allocation end must be rejected");
        };
        match error {
            Error::Format(FormatError::DefinedAddressIsUndefined {
                address,
                offset_size,
            }) if end == 65_535 => assert_eq!((address, offset_size), (end, 2)),
            Error::Format(FormatError::AddressTooLarge {
                address,
                offset_size,
            }) if end == 65_536 => assert_eq!((address, offset_size), (end, 2)),
            error => panic!("expected a width error at {end}, got {error:?}"),
        }
        assert_eq!(store.data, original);
        assert_eq!(store.appends.get(), 0);
        assert_eq!(store.syncs.get(), 0);
        assert!(store.allocations.borrow().is_empty());
        assert!(store.retired.is_empty());
        store.publication.available().unwrap();
        let file = crate::File::from_bytes(store.data).unwrap();
        assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [0, 1, 2]);
        file.close().unwrap();
    }

    #[rstest]
    #[case::widths(true)]
    #[case::creation(false)]
    fn index_initialization_rejects_mismatched_properties_before_allocation(#[case] widths: bool) {
        let mut store = unallocated_index_store(0, false);
        let address = dataset_address(&store);
        let state = edit::locate_dataset_state(&store, address).unwrap();
        let AppendIndex::Unallocated(index) = &state.index else {
            panic!("the fixture must have no index allocation");
        };
        let creation = if widths {
            index.creation
        } else {
            ExtensibleArrayCreationParameters::parse(&[24, 8, 2, 8, 8]).unwrap()
        };
        let encoding_widths = if widths {
            FormatWidths::from_sizes(2, 8).unwrap()
        } else {
            store.widths
        };
        let initialization = ExtensibleArrayInitialization::new(
            creation,
            index.chunk_bytes.to_u64(),
            encoding_widths,
            false,
        )
        .unwrap();
        let original = store.data.clone();
        let Err(error) = initialize_ea_index(&mut store, address, index, &initialization) else {
            panic!("mismatched initialization must be rejected");
        };
        assert!(
            matches!(
                error,
                Error::AppendUnsupported(
                    "index initialization differs from the layout or file widths"
                )
            ),
            "{error:?}"
        );
        assert_eq!(store.data, original);
        assert!(store.allocations.borrow().is_empty());
        assert_eq!(store.syncs.get(), 0);
    }

    #[test]
    fn prepared_allocations_reject_a_different_encoded_length() {
        let mut store = WindowProbeStore::open(build_unlimited(4, 1));
        let original = store.data.clone();
        let Err(error) = store.allocate_encoded(4, AllocationClass::Metadata, |_| Ok([0; 3]))
        else {
            panic!("a different encoded length must be rejected");
        };
        let Error::Format(FormatError::Internal(message)) = error else {
            panic!("expected Internal, got {error:?}");
        };
        assert_eq!(
            message,
            "prepared allocation length differs from its reservation"
        );
        assert_eq!(store.data, original);
        assert!(store.allocations.borrow().is_empty());
    }

    #[rstest]
    #[case::canonical([32, 4, 4, 16, 10])]
    #[case::nondefault([24, 8, 2, 8, 8])]
    fn locating_unallocated_indexes_retains_checked_creation_geometry(
        #[case] parameters: [u8; 5],
        #[values(false, true)] filtered: bool,
    ) {
        let mut store = unallocated_index_store(0, filtered);
        let address = dataset_address(&store);
        let walk = walk_v2_object_header(&store, address.header_address(), store.widths).unwrap();
        let layout = walk
            .messages
            .iter()
            .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
            .unwrap();
        let parameter_start = usize::try_from(layout.data_off).unwrap() + layout.size
            - usize::from(store.widths.offsets.get())
            - parameters.len();
        assert_eq!(
            &store.data[parameter_start..parameter_start + parameters.len()],
            &[32, 4, 4, 16, 10]
        );
        store.data[parameter_start..parameter_start + parameters.len()]
            .copy_from_slice(&parameters);
        test_util::checksum::restamp(
            &mut store.data,
            usize::try_from(layout.chunk_start).unwrap(),
            usize::try_from(layout.chunk_msg_end - layout.chunk_start).unwrap()
                + core::mem::size_of::<u32>(),
        );
        let before = store.data.clone();
        store.reset_counters();
        let state = edit::locate_dataset_state(&store, address).unwrap();
        assert_eq!(state.index.current_dim(), 0);
        assert_eq!(state.index.chunk_elems(), 4);
        assert_eq!(state.index.chunk_bytes(), 16);
        assert_eq!(state.index.element_size().get(), 4);
        assert_eq!(state.pipeline.is_some(), filtered);
        let AppendIndex::Unallocated(index) = state.index else {
            panic!("expected an unallocated index");
        };
        assert_eq!(index.creation.encode(), parameters);
        assert_eq!(store.data, before);
        assert!(store.allocations.borrow().is_empty());
        assert_eq!(store.appends.get(), 0);
        assert_eq!(store.syncs.get(), 0);
    }

    #[rstest]
    fn a_nonempty_unallocated_extent_is_rejected_before_writes(
        #[values(false, true)] filtered: bool,
    ) {
        let store = unallocated_index_store(3, filtered);
        let before = store.data.clone();
        let error = match edit::locate_dataset_state(&store, dataset_address(&store)) {
            Ok(_) => panic!("expected an unsupported nonempty unallocated extent"),
            Err(error) => error,
        };
        let Error::AppendInPlaceUnsupported(reason) = error else {
            panic!("expected AppendInPlaceUnsupported, got {error:?}");
        };
        assert_eq!(
            reason,
            "an unallocated extensible-array index requires an empty dataset"
        );
        assert_eq!(store.data, before);
        assert!(store.allocations.borrow().is_empty());
    }

    fn unallocated_index_store(dimension: u64, filtered: bool) -> WindowProbeStore {
        let mut builder = FileBuilder::new();
        let dataset = builder
            .create_dataset("d")
            .with_unallocated_storage(crate::make_i32_type(), &[dimension])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4]);
        if filtered {
            dataset.with_fletcher32();
        }
        let mut store = WindowProbeStore::open(builder.finish().unwrap());
        clear_layout_index(&mut store);
        store
    }

    fn clear_layout_index(store: &mut WindowProbeStore) {
        let walk =
            walk_v2_object_header(store, dataset_address(store).header_address(), store.widths)
                .unwrap();
        let layout = walk
            .messages
            .into_iter()
            .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
            .unwrap();
        let end = usize::try_from(layout.data_off).unwrap() + layout.size;
        let start = end - usize::from(store.widths.offsets.get());
        store.data[start..end].fill(u8::MAX);
        test_util::checksum::restamp(
            &mut store.data,
            usize::try_from(layout.chunk_start).unwrap(),
            usize::try_from(layout.chunk_msg_end - layout.chunk_start).unwrap()
                + core::mem::size_of::<u32>(),
        );
    }

    #[rstest]
    #[case::rank_two(&[0, 2], &[4, 2], &[MaxExtent::Unlimited, MaxExtent::Fixed(2)], "only rank-1 datasets are supported")]
    fn empty_index_eligibility_rejects_unsupported_dataspaces(
        #[case] shape: &[u64],
        #[case] chunk: &[u64],
        #[case] maximum: &[MaxExtent],
        #[case] expected: &str,
        #[values(false, true)] unallocated: bool,
    ) {
        let mut builder = FileBuilder::new();
        let dataset = builder.create_dataset("d");
        if unallocated {
            dataset.with_unallocated_storage(crate::make_i32_type(), shape);
        } else {
            dataset.with_i32_data(&[]).with_shape(shape);
        }
        dataset.with_chunks(chunk).with_maxshape(maximum);
        let mut store = WindowProbeStore::open(builder.finish().unwrap());
        if unallocated {
            clear_layout_index(&mut store);
        }
        let original = store.data.clone();
        let error = match edit::locate_dataset_state(&store, dataset_address(&store)) {
            Ok(_) => panic!("expected unsupported dataset geometry"),
            Err(error) => error,
        };
        let Error::AppendInPlaceUnsupported(reason) = error else {
            panic!("expected AppendInPlaceUnsupported, got {error:?}");
        };
        assert_eq!(reason, expected);
        assert_eq!(store.data, original);
        assert!(store.allocations.borrow().is_empty());
    }

    #[rstest]
    #[case::direct(4, 54, [0, 0, 3, 80])]
    #[case::secondary(244, 444, [1, 54, 10, 496])]
    fn retry_counts_referenced_blocks_after_interrupted_publication(
        #[case] start: i32,
        #[case] staged_end: i32,
        #[case] counts: [u64; 4],
        #[values(1, 2, 3)] phase: u8,
        #[values(false, true)] reopen: bool,
        #[values(false, true)] filtered: bool,
    ) {
        let mut store = allocation_store(start, filtered);
        let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
        let staged: Vec<_> = (start..staged_end).collect();
        append_state(&mut store, &mut state, &staged, phase).unwrap();
        assert_eq!(
            state.index.allocated().unwrap().allocation_stats.is_none(),
            phase == 2
        );
        if reopen {
            store = WindowProbeStore::open(store.data);
            state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
        }
        append_state(&mut store, &mut state, &[start], 4).unwrap();
        append_state(&mut store, &mut state, &[start + 1], 4).unwrap();
        let [secondary, secondary_bytes, data, capacity] = if phase == 1 {
            if start == 4 {
                [0, 0, 1, 16]
            } else {
                [1, 54, 7, 304]
            }
        } else {
            counts
        };
        assert_eq!(
            stored_stats(&store, state.index.allocated().unwrap()),
            [
                secondary,
                secondary_bytes,
                data,
                capacity * state.index.allocated().unwrap().ea_elem_size.to_u64() + data * 22,
                u64::try_from(start + 2).unwrap(),
                4 + capacity,
            ]
        );
        let file = crate::File::from_bytes(store.data).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_i32().unwrap(),
            (0..start + 2).collect::<Vec<_>>()
        );
    }

    #[rstest]
    fn failed_index_publication_invalidates_the_allocation_snapshot(
        #[values(false, true)] filtered: bool,
    ) {
        let mut store = allocation_store(4, filtered);
        let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
        store.reset_counters();
        store.fail_sync = Some(4);
        let error = append_state(&mut store, &mut state, &[4], 4).unwrap_err();
        let Error::Io(error) = error else {
            panic!("expected the injected I/O failure, got {error:?}");
        };
        assert_eq!(error.kind(), std::io::ErrorKind::Other);
        assert_eq!(error.to_string(), "injected synchronization failure");
        assert!(state.index.allocated().unwrap().allocation_stats.is_none());
        store.fail_sync = None;
        let error = append_state(&mut store, &mut state, &[4], 4).unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{error:?}"
        );
        let mut store = WindowProbeStore::open(store.data);
        let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
        append_state(&mut store, &mut state, &[4], 4).unwrap();
        assert_eq!(
            stored_stats(&store, state.index.allocated().unwrap()),
            [
                0,
                0,
                1,
                16 * state.index.allocated().unwrap().ea_elem_size.to_u64() + 22,
                5,
                20
            ]
        );
        let file = crate::File::from_bytes(store.data).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_i32().unwrap(),
            [0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn allocation_snapshot_is_computed_from_block_pointers() {
        let mut store = allocation_store(4, false);
        let (loc, _) = locate(&store);
        loc.update_ea_header(
            &mut store,
            ExtensibleArrayStats {
                nsuper_blks: u64::MAX,
                super_blk_size: u64::MAX,
                ndata_blks: u64::MAX,
                data_blk_size: u64::MAX,
                max_idx_set: 4,
                nelmts: u64::MAX,
            },
        )
        .unwrap();
        let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
        let before = store.data.clone();
        let plan = state_plan(&store, &state, &[4]).unwrap();
        assert_eq!(store.data, before);
        apply_ea_append(&mut store, state.index.allocated_mut().unwrap(), plan, 4).unwrap();
        assert_eq!(
            stored_stats(&store, state.index.allocated().unwrap()),
            [0, 0, 1, 150, 5, 20]
        );
    }

    #[test]
    fn successful_append_planning_reuses_the_allocation_snapshot() {
        let mut store = allocation_store(50_000, false);
        let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
        append_state(&mut store, &mut state, &[50_000], 4).unwrap();
        let located = state.index.allocated().unwrap();
        let array = metadata_io::read_header(&store, located.ea_addr).unwrap();
        let bytes = IndexBlockBytes::read(&store, &array).unwrap();
        let index = bytes.parse().unwrap();
        let secondary = index
            .super_blocks()
            .map(Result::unwrap)
            .find(|reference| reference.slots().unwrap().contains(&50_001))
            .unwrap();
        let range = secondary.parse_range();
        let start = range.start().get().to_usize().unwrap();
        let end = range.end().get().to_usize().unwrap();
        let secondary = secondary
            .parse(FileRegion::new(range, &store.data[start..end]).unwrap())
            .unwrap();
        let target = secondary
            .data_blocks()
            .map(Result::unwrap)
            .find(|reference| reference.slots().unwrap().contains(&50_001))
            .unwrap();
        let fields = test_util::bytes::signature_offsets(
            &store.data[start..end],
            &target.range().start().get().to_le_bytes(),
        );
        assert_eq!(fields.len(), 1);
        let pointer = (start + fields[0]).to_u64();
        let expected = [(pointer, usize::from(store.widths.offsets.get())); 2];
        store.reset_counters();
        let plan = state_plan(&store, &state, &[50_001]).unwrap();
        let secondary_reads = store
            .reads
            .borrow()
            .iter()
            .copied()
            .filter(|&(offset, _)| range.start().get() <= offset && offset < range.end().get())
            .collect::<Vec<_>>();
        assert_eq!(secondary_reads, expected);
        let cached_reads = store.total_read.get();
        assert!(store.total_read.get() >= 8192);
        assert!(store.total_read.get() <= 16 * 1024);
        assert!(store.max_read.get() <= 1024);
        state.index.allocated_mut().unwrap().allocation_stats = None;
        store.reset_counters();
        let uncached = state_plan(&store, &state, &[50_001]).unwrap();
        assert_eq!(uncached.stats, plan.stats);
        assert!(store.total_read.get() > cached_reads);
        let secondary_reads = store
            .reads
            .borrow()
            .iter()
            .copied()
            .filter(|&(offset, _)| range.start().get() <= offset && offset < range.end().get())
            .collect::<Vec<_>>();
        assert_ne!(secondary_reads, expected);
        apply_ea_append(&mut store, state.index.allocated_mut().unwrap(), plan, 4).unwrap();
        let file = crate::File::from_bytes(store.data).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_i32().unwrap(),
            (0..50_002).collect::<Vec<_>>()
        );
    }

    fn allocation_store(n: i32, filtered: bool) -> WindowProbeStore {
        let mut builder = FileBuilder::new();
        let dataset = builder
            .create_dataset("d")
            .with_i32_data(&(0..n).collect::<Vec<_>>())
            .with_shape(&[u64::try_from(n).unwrap()])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[1]);
        if filtered {
            dataset.with_deflate(DeflateLevel::One);
        }
        WindowProbeStore::open(builder.finish().unwrap())
    }

    fn append_state(
        store: &mut WindowProbeStore,
        state: &mut LocatedState,
        values: &[i32],
        phase: u8,
    ) -> Result<(), Error> {
        let plan = state_plan(store, state, values)?;
        apply_ea_append(store, state.index.allocated_mut().unwrap(), plan, phase)
    }

    fn state_plan(
        store: &WindowProbeStore,
        state: &LocatedState,
        values: &[i32],
    ) -> Result<AppendPlan, Error> {
        plan_ea_append(
            store,
            state.index.allocated().unwrap(),
            &state.datatype,
            &state.spatial,
            state.element_size,
            state.pipeline.as_ref(),
            true,
            LossyTailReencoding::Reject,
            &i32_bytes(values),
            values.len().to_u64(),
            state.fill.pattern(state.element_size),
        )
    }

    fn stored_stats(store: &WindowProbeStore, loc: &Located) -> [u64; EA_STAT_FIELDS] {
        store
            .read_exact_at(loc.ea_addr.get() + EA_STATS_OFFSET, EA_STAT_FIELDS * 8)
            .unwrap()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|field| u64::from_le_bytes(*field))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    #[rstest]
    #[case::first_direct(4, 4, 16, 1, 0, None, 0)]
    #[case::last_in_first_direct(19, 4, 16, 1, 0, None, 0)]
    #[case::next_direct(20, 20, 32, 1, 0, None, 1)]
    #[case::first_secondary(244, 244, 64, 4, 240, Some(0), 0)]
    #[case::last_in_secondary(499, 436, 64, 4, 240, Some(0), 3)]
    #[case::next_secondary(500, 500, 128, 4, 496, Some(1), 0)]
    #[case::first_paged(131_060, 131_060, 2_048, 64, 131_056, Some(9), 0)]
    #[case::next_paged_block(133_108, 133_108, 2_048, 64, 131_056, Some(9), 1)]
    fn data_block_locations_match_expected_ranges(
        #[case] element: u64,
        #[case] start: u64,
        #[case] elements: u64,
        #[case] blocks: u64,
        #[case] offset: u64,
        #[case] secondary: Option<usize>,
        #[case] local: usize,
    ) {
        let header = range_header(32, 4, 16);
        let geom = ExtensibleArrayGeometry::from_header(&header, OffsetWidth::Eight).unwrap();
        let region = locate_data_block(&geom, 4, element).unwrap();
        assert_eq!(
            (
                region.db_start,
                region.dblk_nelmts,
                region.ndblks,
                region.sb_block_offset
            ),
            (start, elements, blocks, offset)
        );
        match (region.parent, secondary) {
            (Parent::IndexDirect { ordinal }, None) => assert_eq!(ordinal, local),
            (Parent::Super { sblk_j, dblk_local }, Some(expected)) => {
                assert_eq!((sblk_j, dblk_local), (expected, local));
            }
            _ => panic!("unexpected parent block"),
        }
    }

    #[rstest]
    #[case::span(64, 4, 16, u64::MAX - 11)]
    #[case::endpoint(63, 16, 16, u64::MAX)]
    fn data_block_range_overflow_is_rejected(
        #[case] bits: u8,
        #[case] inline: u8,
        #[case] minimum: u8,
        #[case] element: u64,
    ) {
        let header = range_header(bits, inline, minimum);
        let geom = ExtensibleArrayGeometry::from_header(&header, OffsetWidth::Eight).unwrap();
        let inline = u64::from(inline);
        let error = match locate_data_block(&geom, inline, element) {
            Ok(_) => panic!("expected a range error"),
            Err(error) => error,
        };
        assert!(
            matches!(
                error,
                Error::Format(FormatError::InvalidExtensibleArrayGeometry)
            ),
            "{error:?}"
        );
    }

    #[test]
    fn the_last_representable_secondary_range_has_the_expected_location() {
        let header = range_header(64, 4, 16);
        let geom = ExtensibleArrayGeometry::from_header(&header, OffsetWidth::Eight).unwrap();
        let end = (1u128 << 64) - 12;
        let elements = 1u64 << 34;
        let region = locate_data_block(&geom, 4, u64::try_from(end - 1).unwrap()).unwrap();
        assert_eq!(u128::from(region.db_start) + u128::from(elements), end);
        assert_eq!(region.dblk_nelmts, elements);
        assert_eq!(region.ndblks, 1 << 29);
        assert_eq!(region.sb_block_offset, (1 << 63) - 16);
        let Parent::Super { sblk_j, dblk_local } = region.parent else {
            panic!("expected a secondary block");
        };
        assert_eq!((sblk_j, dblk_local), (55, (1 << 29) - 1));
    }

    #[rstest]
    #[case::inline(4, 0, 0, 0, 4)]
    #[case::first_direct(5, 0, 0, 1, 20)]
    #[case::secondary(245, 1, 54, 7, 308)]
    fn dense_prefix_statistics_match_expected_ranges(
        #[case] chunks: u64,
        #[case] secondary_count: u64,
        #[case] secondary_bytes: u64,
        #[case] data_count: u64,
        #[case] elements: u64,
        #[values(8, 14)] stride: usize,
    ) {
        let store = WindowProbeStore::open(build_unlimited(4, 1));
        let (mut loc, _) = locate(&store);
        loc.ea_elem_size = stride;
        assert_eq!(
            loc.dense_prefix_stats(chunks).unwrap(),
            ExtensibleArrayStats {
                nsuper_blks: secondary_count,
                super_blk_size: secondary_bytes,
                ndata_blks: data_count,
                data_blk_size: (elements - 4) * stride.to_u64() + data_count * 22,
                max_idx_set: chunks,
                nelmts: elements,
            }
        );
    }

    #[rstest]
    #[case::span(64, u64::MAX - 11)]
    #[cfg_attr(target_pointer_width = "64", case::statistics(63, 1u64 << 61))]
    fn append_planning_rejects_unrepresentable_ranges_before_reading(
        #[case] bits: u8,
        #[case] dimension: u64,
    ) {
        let store = WindowProbeStore::open(build_unlimited(4, 1));
        let original = store.data.clone();
        let (mut loc, datatype) = locate(&store);
        let header = range_header(bits, 4, 16);
        loc.geom = ExtensibleArrayGeometry::from_header(&header, OffsetWidth::Eight).unwrap();
        loc.ea_bits = header.bit_geometry();
        loc.current_dim = dimension;
        store.reset_counters();
        let error = match plan_ea_append(
            &store,
            &loc,
            &datatype,
            &[1],
            loc.elem_bytes,
            None,
            true,
            LossyTailReencoding::Reject,
            &9i32.to_le_bytes(),
            1,
            FillPattern::ZERO,
        ) {
            Ok(_) => panic!("expected a range error"),
            Err(error) => error,
        };
        assert!(
            matches!(
                error,
                Error::Format(FormatError::InvalidExtensibleArrayGeometry)
            ),
            "{error:?}"
        );
        assert_eq!(store.total_read.get(), 0);
        assert_eq!(store.allocations.borrow().as_slice(), []);
        assert_eq!(store.data, original);
    }

    fn range_header(bits: u8, inline: u8, minimum: u8) -> ExtensibleArrayHeader {
        let mut bytes = EA_HEADER_SIGNATURE.to_vec();
        bytes.extend_from_slice(&[
            EA_HEADER_VERSION,
            EA_CLIENT_ID_UNFILTERED,
            8,
            bits,
            inline,
            minimum,
            4,
            10,
        ]);
        bytes.resize(ExtensibleArrayHeader::serialized_size(8, 8), 0);
        let len = bytes.len();
        checksum::stamp_trailing(&mut bytes, 0, len);
        ExtensibleArrayHeader::parse(&bytes, 0, 8, 8).unwrap()
    }

    #[rstest]
    #[case::inline(1, 2, false, false)]
    #[case::direct_data_block(4, 5, true, false)]
    #[case::super_block(4, 324, true, true)]
    fn append_distinguishes_chunk_and_index_allocation_intent(
        #[case] start: i32,
        #[case] end: i32,
        #[case] data_block: bool,
        #[case] super_block: bool,
    ) {
        let mut store = WindowProbeStore::open(build_unlimited(start, 1));
        let (mut loc, datatype) = locate(&store);
        store.reset_counters();
        alloc_probe::take();
        append_i32s(&mut store, &mut loc, &datatype, start..end);
        let (data_blocks, super_blocks) = alloc_probe::take();
        assert_eq!(
            (data_blocks > 0, super_blocks > 0),
            (data_block, super_block)
        );
        let mut expected = vec![AllocationClass::Raw; (end - start) as usize];
        expected.extend(core::iter::repeat_n(
            AllocationClass::Metadata,
            data_blocks + super_blocks,
        ));
        assert_eq!(*store.allocations.borrow(), expected);
        assert_eq!(read_i32s(&store, &loc), (0..end).collect::<Vec<_>>());
        store.reset_counters();
        assert_eq!(*store.allocations.borrow(), []);
    }

    /// An in-test [`Store`] over a `Vec<u8>` that records every read window, so
    /// the tests below can assert the engine's bounded-memory contract: the
    /// append path never reads more than a bounded window at once, no matter how
    /// large the file is. This is the seam contract issue #147's mirror-less
    /// bounded backend relies on.
    struct WindowProbeStore {
        data: Vec<u8>,
        publication: AllocationPublication,
        superblock: Superblock,
        widths: FormatWidths,
        sb_sig_off: usize,
        max_read: Cell<usize>,
        total_read: Cell<usize>,
        reads: RefCell<Vec<(u64, usize)>>,
        writes: Vec<(u64, usize)>,
        allocations: RefCell<Vec<AllocationClass>>,
        retired: Vec<FileRange>,
        /// How many times the append sequence rewrote the superblock to advance
        /// its recorded end-of-file. Two of the four durability phases can, and
        /// only one of them always must.
        superblock_patches: Cell<usize>,
        /// Barriers the append sequence issued. Every one is a point where the
        /// writes before it must reach the disk before the writes after it.
        syncs: Cell<usize>,
        /// Blobs appended at end-of-file: the chunk, plus one per fresh index
        /// block. Each fresh block owes a barrier before the pointer naming it.
        appends: Cell<usize>,
        /// A free region within the recorded file length, created by
        /// [`with_free_tail`](Self::with_free_tail). The store allocates from it before appending.
        free: Option<(u64, u64)>,
        fail_sync: Option<usize>,
        /// The number of allocations the store served from `free`.
        reuses: Cell<usize>,
    }

    impl WindowProbeStore {
        fn open(data: Vec<u8>) -> Self {
            let LocatedSuperblock {
                address,
                superblock,
                widths,
            } = crate::superblock::read(data.as_slice()).unwrap();
            let sb_sig_off = usize::try_from(address.get()).unwrap();
            Self {
                data,
                publication: AllocationPublication::new(),
                superblock,
                widths,
                sb_sig_off,
                max_read: Cell::new(0),
                total_read: Cell::new(0),
                reads: RefCell::new(Vec::new()),
                writes: Vec::new(),
                allocations: RefCell::new(Vec::new()),
                retired: Vec::new(),
                superblock_patches: Cell::new(0),
                syncs: Cell::new(0),
                appends: Cell::new(0),
                free: None,
                fail_sync: None,
                reuses: Cell::new(0),
            }
        }

        /// Give the store a `len`-byte free region, by growing the image and
        /// recording the new end-of-file in the superblock before any counting
        /// starts. That gives the region the two properties the engine can
        /// observe — inside the recorded end-of-file, and named by nothing.
        ///
        /// It sits at the tail, where a *committed* deletion never leaves one
        /// (a free run reaching end-of-file is truncated away instead, which is
        /// why the integration tests plant a ceiling object above theirs). The
        /// engine cannot tell the difference: it reads an address and a length.
        fn with_free_tail(mut self, len: u64) -> Self {
            let addr = self.data.len() as u64;
            self.data
                .resize(self.data.len() + len.to_usize().unwrap(), 0);
            self.patch_superblock_eof().unwrap();
            self.free = Some((addr, len));
            self
        }

        /// Extends the image when the free region cannot fit an allocation.
        fn append_bytes(&mut self, bytes: &[u8]) -> Result<u64, Error> {
            let addr = self.data.len() as u64;
            self.require_allocation_range(addr, bytes.len().to_u64())?;
            self.appends.set(self.appends.get() + 1);
            self.data.extend_from_slice(bytes);
            Ok(addr)
        }

        fn require_allocation_range(&self, address: u64, length: u64) -> Result<(), Error> {
            let range = FileRange::new(PhysicalAddress::new(address), length)?;
            let mut encoded = [0; 8];
            hdf5_pure_format::__private::write_defined_stored_address_at(
                &mut encoded,
                0,
                self.addresses().relative(range.start())?,
                self.widths.offsets,
            )?;
            hdf5_pure_format::__private::write_defined_stored_address_at(
                &mut encoded,
                0,
                self.addresses().relative(range.end())?,
                self.widths.offsets,
            )?;
            Ok(())
        }

        fn allocate_bytes(&mut self, bytes: &[u8], class: AllocationClass) -> Result<u64, Error> {
            let len = bytes.len() as u64;
            let address = self
                .free
                .filter(|&(_, available)| available >= len)
                .map_or(self.data.len().to_u64(), |(address, _)| address);
            self.require_allocation_range(address, len)?;
            self.allocations.get_mut().push(class);
            if let Some((addr, avail)) = self.free.filter(|&(_, avail)| avail >= len) {
                self.require_allocation_range(addr, len)?;
                self.reuses.set(self.reuses.get() + 1);
                self.free = (avail > len).then(|| (addr + len, avail - len));
                let start = addr.to_usize()?;
                self.data[start..start + bytes.len()].copy_from_slice(bytes);
                return Ok(addr);
            }
            self.append_bytes(bytes)
        }
        fn patch_superblock_eof(&mut self) -> Result<(), Error> {
            self.superblock_patches
                .set(self.superblock_patches.get() + 1);
            self.superblock.eof_address = self.data.len() as u64;
            let bytes =
                hdf5_pure_format::__private::serialize_superblock(&self.superblock).unwrap();
            let off = self.sb_sig_off;
            self.data[off..off + bytes.len()].copy_from_slice(&bytes);
            Ok(())
        }

        fn reset_counters(&self) {
            self.max_read.set(0);
            self.total_read.set(0);
            self.reads.borrow_mut().clear();
            self.allocations.borrow_mut().clear();
            self.superblock_patches.set(0);
            self.syncs.set(0);
            self.appends.set(0);
            self.reuses.set(0);
        }
    }

    impl Source for WindowProbeStore {
        fn len(&self) -> u64 {
            self.data.len() as u64
        }
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            self.max_read.set(self.max_read.get().max(buf.len()));
            self.total_read.set(self.total_read.get() + buf.len());
            self.reads.borrow_mut().push((offset, buf.len()));
            BytesSource::new(&self.data).read_at(offset, buf)
        }
    }

    impl Store for WindowProbeStore {
        fn opened_source(&self) -> OpenedSource<'_> {
            OpenedSource::for_test(self, self.addresses())
        }
        fn widths(&self) -> FormatWidths {
            self.widths
        }
        fn addresses(&self) -> AddressContext {
            AddressContext::from_effective_base(self.superblock.base_address)
        }
        fn logical_end(&self) -> u64 {
            self.superblock.eof_address
        }
        fn require_exclusive_chunk_storage(&self, _header: ObjectId) -> Result<(), Error> {
            Ok(())
        }
        fn retire(&mut self, retirement: RetiredChunk) {
            self.retired.push(retirement.range());
        }
        fn allocation_publication(&self) -> &AllocationPublication {
            &self.publication
        }
        fn allocate(
            &mut self,
            bytes: &[u8],
            class: AllocationClass,
        ) -> Result<ProvisionalAllocation, Error> {
            self.publication.available()?;
            let address = self.allocate_bytes(bytes, class)?;
            self.publication.provision(FileRange::new(
                PhysicalAddress::new(address),
                bytes.len() as u64,
            )?)
        }
        fn allocate_encoded<B: AsRef<[u8]>>(
            &mut self,
            length: u64,
            class: AllocationClass,
            prepare: impl FnOnce(PhysicalAddress) -> Result<B, Error>,
        ) -> Result<ProvisionalAllocation, Error> {
            self.publication.available()?;
            let address = self
                .free
                .filter(|&(_, available)| available >= length)
                .map_or(self.data.len().to_u64(), |(address, _)| address);
            self.require_allocation_range(address, length)?;
            let bytes = prepare(PhysicalAddress::new(address))?;
            if bytes.as_ref().len().to_u64() != length {
                return Err(FormatError::Internal(
                    "prepared allocation length differs from its reservation".into(),
                )
                .into());
            }
            self.allocate(bytes.as_ref(), class)
        }

        fn allocate_emitted(
            &mut self,
            length: NonZeroByteLength,
            class: AllocationClass,
            emit: impl FnOnce(
                PhysicalAddress,
                &mut dyn FnMut(&[u8]) -> Result<(), Error>,
            ) -> Result<(), Error>,
        ) -> Result<ProvisionalAllocation, Error> {
            self.allocate_encoded(length.get(), class, |address| {
                let mut output = Vec::new();
                let mut rejected = false;
                emit(address, &mut |bytes| {
                    if rejected {
                        return Err(FormatError::Internal(
                            "reservation emission has an earlier failed write".into(),
                        )
                        .into());
                    }
                    let remaining = length.get() - output.len().to_u64();
                    if bytes.len().to_u64() > remaining {
                        rejected = true;
                        return Err(FormatError::SerializationError(format!(
                            "a placement reserved {} bytes but emitted more than its remaining {remaining}",
                            length.get(),
                        ))
                        .into());
                    }
                    output.extend_from_slice(bytes);
                    Ok(())
                })?;
                if rejected {
                    return Err(FormatError::Internal(
                        "reservation emission has an earlier failed write".into(),
                    )
                    .into());
                }
                if output.len().to_u64() != length.get() {
                    return Err(FormatError::SerializationError(format!(
                        "a placement reserved {} bytes but emitted {}",
                        length.get(),
                        output.len(),
                    ))
                    .into());
                }
                Ok(output)
            })
        }

        fn make_allocation_uncertain(&mut self) {
            self.publication.make_uncertain();
        }

        fn publish_allocations(&mut self) -> Result<ConfirmedPublication, Error> {
            let attempt = match self.publication.begin()? {
                PublicationAction::Current(publication) => {
                    return Ok(publication);
                }
                PublicationAction::Publish(attempt) => attempt,
            };
            self.sync()?;
            if self.data.len() as u64 != self.superblock.eof_address {
                self.patch_superblock_eof()?;
                self.sync()?;
            }
            self.publication.complete(attempt)
        }
        fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
            self.publication.before_reference()?;
            self.writes.push((offset, bytes.len()));
            let offset = offset.to_usize()?;
            self.data[offset..offset + bytes.len()].copy_from_slice(bytes);
            Ok(())
        }
        fn sync(&mut self) -> Result<(), Error> {
            self.syncs.set(self.syncs.get() + 1);
            if self.fail_sync == Some(self.syncs.get()) {
                self.publication.make_uncertain();
                return Err(Error::Io(std::io::Error::other(
                    "injected synchronization failure",
                )));
            }
            Ok(())
        }
    }

    /// Build an in-memory latest-format file with one unlimited chunked i32
    /// dataset seeded with `0..n`, returning its bytes.
    fn build_unlimited(n: i32, chunk: u64) -> Vec<u8> {
        let data: Vec<i32> = (0..n).collect();
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&data)
            .with_shape(&[n as u64])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[chunk]);
        b.finish().unwrap()
    }

    /// Drive the full located-append path (`locate_at` + `plan_ea_append` +
    /// `apply_ea_append`) through the probe store and return the located state.
    fn locate(store: &WindowProbeStore) -> (Located, crate::datatype::Datatype) {
        let result =
            Located::locate_at(store, dataset_address(store), Error::AppendUnsupported).unwrap();
        (result.located.into_allocated().unwrap(), result.datatype)
    }

    fn dataset_address(store: &WindowProbeStore) -> ObjectId {
        group_v2::resolve_path_any(
            PhysicalBytes::for_test(
                &store.data,
                AddressContext::from_effective_base(store.superblock.base_address),
            ),
            AccessMode::ReadOnly,
            &store.superblock,
            store.widths,
            &ObjectPath::parse("d"),
        )
        .map(PhysicalAddress::new)
        .map(ObjectId::new)
        .unwrap()
    }

    /// Every element of the located dataset, read back through the engine's own
    /// index walk: one unfiltered chunk of `chunk_elems` i32 per indexed slot.
    fn read_i32s(store: &WindowProbeStore, loc: &Located) -> Vec<i32> {
        let width = loc.chunk_elems.to_usize().unwrap() * 4;
        let mut out = Vec::new();
        for e in 0..loc.num_chunks {
            let rec = loc.read_element(store, e).unwrap().expect("indexed chunk");
            let bytes = store.read_exact_at(rec.addr.get(), width).unwrap();
            out.extend(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .copied()
                    .map(i32::from_le_bytes),
            );
        }
        out
    }

    fn append_i32s(
        store: &mut WindowProbeStore,
        loc: &mut Located,
        datatype: &crate::datatype::Datatype,
        values: std::ops::Range<i32>,
    ) {
        let raw: Vec<u8> = values.clone().flat_map(|v| v.to_le_bytes()).collect();
        let new_elems = (values.end - values.start) as u64;
        let spatial = vec![loc.chunk_elems];
        let plan = plan_ea_append(
            store,
            loc,
            datatype,
            &spatial,
            loc.elem_bytes,
            None,
            true,
            LossyTailReencoding::Reject,
            &raw,
            new_elems,
            FillPattern::ZERO,
        )
        .unwrap();
        apply_ea_append(store, loc, plan, 4).unwrap();
    }

    /// Returns the bytes of the file from a fresh data block of `dblk_nelmts`
    /// elements on, and the prefix that block opens with.
    fn fresh_data_block(dblk_nelmts: u64) -> (Vec<u8>, Vec<u8>) {
        let mut store = WindowProbeStore::open(build_unlimited(4, 1));
        let (loc, _) = locate(&store);
        let at = loc
            .alloc_undef_data_block(&mut store, dblk_nelmts, 32)
            .unwrap()
            .range()
            .start()
            .get()
            .to_usize()
            .unwrap();
        let mut prefix = b"EADB".to_vec();
        prefix.push(0);
        prefix.push(loc.client_id);
        prefix.extend_from_slice(&loc.ea_addr.get().to_le_bytes());
        prefix.extend_from_slice(&32u64.to_le_bytes()[..loc.ea_bits.block_offset_width().bytes()]);
        (store.data[at..].to_vec(), prefix)
    }

    #[test]
    fn the_header_walk_reads_the_prefix_window_then_the_chunk_0_messages() {
        let store = WindowProbeStore::open(build_unlimited(4, 1));
        let oh_addr = group_v2::resolve_path_any(
            PhysicalBytes::for_test(
                &store.data,
                AddressContext::from_effective_base(store.superblock.base_address),
            ),
            AccessMode::ReadOnly,
            &store.superblock,
            crate::file_writer::WIDTHS,
            &ObjectPath::parse("d"),
        )
        .unwrap();
        store.reset_counters();

        let walk =
            walk_v2_object_header(&store, PhysicalAddress::new(oh_addr), store.widths).unwrap();

        let reads = store.reads.borrow().clone();
        let [(head_at, head_len), (messages_at, messages_len)] = reads[..] else {
            panic!("expected the prefix window and the chunk 0 messages, got {reads:?}");
        };
        assert_eq!((head_at, head_len), (oh_addr, OBJECT_HEADER_PREFIX_MAX_LEN));
        let record_prefix = MessageRecordLayout::PLAIN.prefix_len().to_u64();
        assert_eq!(
            walk.messages
                .iter()
                .map(|message| (message.msg_type, message.chunk_msg_end))
                .collect::<Vec<_>>(),
            [
                MessageType::DATATYPE,
                MessageType::DATASPACE,
                MessageType::FILL_VALUE,
                MessageType::DATA_LAYOUT,
            ]
            .map(|msg_type| (msg_type, messages_at + messages_len.to_u64()))
        );
        assert_eq!(walk.messages[0].data_off, messages_at + record_prefix);
    }

    #[rstest::rstest]
    #[case::a_version_other_than_2(header_of_version_1())]
    #[case::a_prefix_cut_short(header_cut_inside_its_prefix())]
    #[case::a_chunk_0_past_the_end_of_the_file(chunk_0_past_the_end_of_the_file())]
    #[case::a_record_past_the_end_of_chunk_0(record_past_the_end_of_chunk_0())]
    #[case::a_short_continuation_message(continuation_message_shorter_than_its_fields())]
    #[case::a_continuation_to_the_undefined_address(continuation_message_to_the_undefined_address())]
    #[case::a_short_continuation_block(continuation_block_shorter_than_its_signature_and_checksum())]
    #[case::a_continuation_block_without_its_signature(continuation_block_without_its_signature())]
    fn a_damaged_header_fails_the_walk(#[case] (bytes, expected): (Vec<u8>, FormatError)) {
        let Err(err) = walk_v2_object_header(
            &BytesSource::new(&bytes),
            PhysicalAddress::new(0),
            crate::file_writer::WIDTHS,
        ) else {
            panic!("expected the walk to fail with {expected:?}");
        };
        let Error::Format(format_error) = err else {
            panic!("expected a format error, got {err:?}");
        };
        assert_eq!(format_error, expected);
    }

    fn header_of_version_1() -> (Vec<u8>, FormatError) {
        let mut header = one_message_header().build();
        header[4] = 1;
        (header, FormatError::InvalidObjectHeaderVersion(1))
    }

    fn header_cut_inside_its_prefix() -> (Vec<u8>, FormatError) {
        let header = one_message_header().build()[..5].to_vec();
        (
            header,
            FormatError::UnexpectedEof {
                expected: 6,
                available: 5,
            },
        )
    }

    fn chunk_0_past_the_end_of_the_file() -> (Vec<u8>, FormatError) {
        let mut header = one_message_header()
            .flags(v2_bytes::HeaderFlags::CHUNK_SIZE_WIDTH)
            .build();
        header[6..14].copy_from_slice(&(1u64 << 31).to_le_bytes());
        let available = header.len();
        (
            header,
            FormatError::UnexpectedSourceEof {
                expected: 14 + (1 << 31),
                available: available.to_u64(),
            },
        )
    }

    fn record_past_the_end_of_chunk_0() -> (Vec<u8>, FormatError) {
        let mut header = one_message_header().build();
        header[8] = 0xFF;
        (
            header,
            FormatError::UnexpectedEof {
                expected: 4 + 0xFF,
                available: 8,
            },
        )
    }

    fn continuation_message_shorter_than_its_fields() -> (Vec<u8>, FormatError) {
        let header = v2_bytes::Header::new()
            .message(Message::new(
                RecordType::OBJECT_HEADER_CONTINUATION,
                &[0; 4],
            ))
            .build();
        (
            header,
            FormatError::UnexpectedEof {
                expected: 8,
                available: 4,
            },
        )
    }

    fn continuation_message_to_the_undefined_address() -> (Vec<u8>, FormatError) {
        let header = v2_bytes::Header::new()
            .message(Message::undefined_continuation(64, Widths::EIGHT))
            .build();
        (header, FormatError::UndefinedContinuationAddress)
    }

    fn continuation_block_shorter_than_its_signature_and_checksum() -> (Vec<u8>, FormatError) {
        (
            image_with_continuation_block(
                &[v2_bytes::CONTINUATION_SIGNATURE.as_slice(), &[0; 3]].concat(),
            ),
            FormatError::InvalidObjectHeaderSignature,
        )
    }

    fn continuation_block_without_its_signature() -> (Vec<u8>, FormatError) {
        (
            image_with_continuation_block(&[v2_bytes::SIGNATURE.as_slice(), &[0; 4]].concat()),
            FormatError::InvalidObjectHeaderSignature,
        )
    }

    fn one_message_header() -> v2_bytes::Header {
        v2_bytes::Header::new().message(Message::new(RecordType::DATASPACE, &[1, 2, 3, 4]))
    }

    fn image_with_continuation_block(block: &[u8]) -> Vec<u8> {
        const BLOCK_AT: usize = 0x100;
        let header = v2_bytes::Header::new()
            .continuation(BLOCK_AT, block, Widths::EIGHT)
            .build();
        let mut image = Image::starting_with(&header);
        image.place(BLOCK_AT, block);
        image.build()
    }

    #[rstest]
    #[case::zero_width(0)]
    #[case::below_address_width(7)]
    #[case::above_address_width(9)]
    #[case::maximum_width(255)]
    fn an_unfiltered_element_width_other_than_an_address_is_readable_but_append_is_unsupported(
        #[case] element_size: u8,
    ) {
        let mut data = build_unlimited(4, 1);
        let header_at = data
            .windows(4)
            .position(|window| window == b"EAHD")
            .unwrap();
        data[header_at + 6] = element_size;
        test_util::checksum::restamp(
            &mut data,
            header_at,
            ExtensibleArrayHeader::serialized_size(8, 8),
        );
        assert_eq!(
            crate::File::from_bytes(data.clone())
                .unwrap()
                .dataset("d")
                .unwrap()
                .read_i32()
                .unwrap(),
            [0, 1, 2, 3]
        );
        let store = WindowProbeStore::open(data);
        let oh_addr = group_v2::resolve_path_any(
            PhysicalBytes::for_test(
                &store.data,
                AddressContext::from_effective_base(store.superblock.base_address),
            ),
            AccessMode::ReadOnly,
            &store.superblock,
            crate::file_writer::WIDTHS,
            &ObjectPath::parse("d"),
        )
        .unwrap();

        let Err(err) = Located::locate_at(
            &store,
            ObjectId::new(PhysicalAddress::new(oh_addr)),
            Error::AppendUnsupported,
        ) else {
            panic!("expected an unsupported element width");
        };
        let Error::AppendUnsupported(reason) = err else {
            panic!("expected AppendUnsupported, got {err:?}");
        };
        assert_eq!(
            reason,
            "malformed unfiltered extensible-array element width"
        );
    }

    #[rstest]
    fn locating_an_array_reads_its_complete_header_once(#[values(false, true)] filtered: bool) {
        let store = if filtered {
            EdgeFilter::Shuffle.store()
        } else {
            WindowProbeStore::open(build_unlimited(4, 1))
        };
        let address = dataset_address(&store);
        store.reset_counters();
        let located = Located::locate_at(&store, address, Error::AppendUnsupported)
            .unwrap()
            .located
            .into_allocated()
            .unwrap();
        assert_eq!(
            store
                .reads
                .borrow()
                .iter()
                .filter(|(at, _)| *at == located.ea_addr.get())
                .copied()
                .collect::<Vec<_>>(),
            [(
                located.ea_addr.get(),
                ExtensibleArrayHeader::serialized_size_with_widths(store.widths),
            )]
        );
    }

    #[test]
    fn a_filtered_header_with_an_empty_size_field_parses_and_is_unsupported_for_append() {
        let mut store = EdgeFilter::Shuffle.store();
        let address = dataset_address(&store);
        let header = Located::locate_at(&store, address, Error::AppendUnsupported)
            .unwrap()
            .located
            .into_allocated()
            .unwrap()
            .ea_addr
            .get()
            .to_usize()
            .unwrap();
        store.data[header + EA_HEADER_ELEMENT_SIZE] =
            store.widths.offsets.get() + EA_FILTER_MASK_BYTES;
        test_util::checksum::restamp(
            &mut store.data,
            header,
            ExtensibleArrayHeader::serialized_size_with_widths(store.widths),
        );
        let array = crate::extensible_array::read_header(
            &source::zero_base_source(&store),
            store.widths,
            StoredAddress::new(header.to_u64()),
        )
        .unwrap();
        assert_eq!(
            array.header().element_size,
            store.widths.offsets.get() + EA_FILTER_MASK_BYTES
        );
        let Err(error) = Located::locate_at(&store, address, Error::AppendUnsupported) else {
            panic!("expected an unsupported element width");
        };
        let Error::AppendUnsupported(reason) = error else {
            panic!("expected AppendUnsupported, got {error:?}");
        };
        assert_eq!(reason, "malformed filtered extensible-array element width");
    }

    #[rstest]
    #[case::zero_chunk(0, "chunk dimensions must be nonzero")]
    #[case::chunk_byte_overflow(u64::MAX, "chunk logical byte size exceeds the addressable range")]
    fn immediate_append_rejects_zero_or_overflowing_chunk_geometry(
        #[case] chunk_elems: u64,
        #[case] expected: &str,
    ) {
        let mut store = WindowProbeStore::open(build_unlimited(6, 4));
        let walk = walk_v2_object_header(
            &store,
            dataset_address(&store).header_address(),
            store.widths,
        )
        .unwrap();
        let mut header = v2_bytes::Header::new();
        for msg_type in [
            MessageType::DATASPACE,
            MessageType::DATATYPE,
            MessageType::DATA_LAYOUT,
        ] {
            let message = walk
                .messages
                .iter()
                .find(|msg| msg.msg_type == msg_type)
                .unwrap();
            let mut body = store
                .read_metadata_at(message.data_off, message.size)
                .unwrap();
            if msg_type == MessageType::DATA_LAYOUT {
                let dimensions = body[V4_DIMENSION_COUNT_OFFSET] as usize;
                let old_width = body[V4_DIMENSION_WIDTH_OFFSET] as usize;
                let suffix = body[V4_DIMENSIONS_OFFSET + dimensions * old_width..].to_vec();
                body.truncate(V4_DIMENSIONS_OFFSET);
                body[V4_DIMENSION_WIDTH_OFFSET] = FIXTURE_DIMENSION_BYTES;
                body.extend_from_slice(&chunk_elems.to_le_bytes());
                body.extend_from_slice(&4_u64.to_le_bytes());
                body.extend_from_slice(&suffix);
            }
            header = header.message(Message::new(RecordType(msg_type.to_u16()), &body));
        }
        let at = store.data.len() as u64;
        store.data.extend(header.build());
        store.patch_superblock_eof().unwrap();
        let original = store.data.clone();
        let Err(error) =
            edit::locate_dataset_state(&store, ObjectId::new(PhysicalAddress::new(at)))
        else {
            panic!("expected invalid chunk geometry");
        };
        let Error::Format(FormatError::InvalidChunkGeometry(reason)) = error else {
            panic!("expected InvalidChunkGeometry, got {error:?}");
        };
        assert_eq!(reason, expected);
        assert_eq!(store.data, original);
    }

    #[rstest]
    #[case::unfiltered(false)]
    #[case::filtered(true)]
    fn locating_append_state_reads_the_datatype_once(#[case] filtered: bool) {
        let store = if filtered {
            EdgeFilter::Shuffle.store()
        } else {
            WindowProbeStore::open(build_unlimited(6, 4))
        };
        let at = dataset_address(&store);
        let walk = walk_v2_object_header(&store, at.header_address(), store.widths).unwrap();
        let datatype = walk
            .messages
            .iter()
            .find(|msg| msg.msg_type == MessageType::DATATYPE)
            .unwrap();
        store.reset_counters();
        let state = edit::locate_dataset_state(&store, at).unwrap();
        assert_eq!(state.datatype, crate::type_builders::make_i32_type());
        assert_eq!(
            store
                .reads
                .borrow()
                .iter()
                .filter(|&&(at, len)| at == datatype.data_off && len == datatype.size)
                .count(),
            1,
        );
    }

    #[test]
    fn a_layout_element_size_other_than_the_datatype_size_is_rejected() {
        let mut store = WindowProbeStore::open(build_unlimited(6, 4));
        let oh_addr = crate::File::from_bytes(store.data.clone())
            .unwrap()
            .dataset("d")
            .unwrap()
            .header_address()
            .unwrap();
        let walk =
            walk_v2_object_header(&store, PhysicalAddress::new(oh_addr), store.widths).unwrap();
        let datatype = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DATATYPE)
            .unwrap();
        let wider =
            hdf5_pure_format::__private::serialize_datatype(&crate::type_builders::make_i64_type())
                .unwrap();
        assert_eq!(wider.len(), datatype.size);
        store
            .publish_checksummed(
                datatype.chunk_start,
                datatype.chunk_msg_end,
                datatype.data_off,
                &wider,
            )
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("element_size.h5");
        std::fs::write(&path, &store.data).unwrap();

        let file = crate::File::open_rw(&path).unwrap();
        let err = file.dataset("d").unwrap().append(&[6_i64]).unwrap_err();
        let Error::Format(FormatError::ChunkElementSizeMismatch { datatype, layout }) = err else {
            panic!("expected ChunkElementSizeMismatch, got {err:?}");
        };
        assert_eq!((datatype, layout), (8, 4));
        file.close().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), store.data);
    }

    #[test]
    fn a_fresh_unpaged_data_block_holds_only_undefined_elements() {
        let (block, mut expected) = fresh_data_block(16);
        expected.extend_from_slice(&[0xFF; 16 * 8]);
        test_util::checksum::append(&mut expected);
        assert_eq!(block[..expected.len()], expected);
    }

    #[test]
    fn a_fresh_paged_data_block_checksums_its_prefix_and_each_page() {
        let (block, mut expected) = fresh_data_block(2048);
        test_util::checksum::append(&mut expected);
        let mut page = vec![0xFF; 1024 * 8];
        test_util::checksum::append(&mut page);
        expected.extend_from_slice(&page);
        expected.extend_from_slice(&page);
        assert_eq!(block[..expected.len()], expected);
    }

    /// A fresh Extensible-Array block is separated from the pointer that names
    /// it by a barrier, so the block is on the disk before anything reaches it.
    ///
    /// The block's bytes are allocated fresh; the slot that points at it lives in
    /// the index block or a super block, near the front of the file. Left in one
    /// barrier-free window, an image that gathers its writes issues them in
    /// *address* order — the pointer first — and a failure in between leaves an
    /// index block whose checksum validates and whose pointer names bytes past
    /// the end of the file. The next append then reads that pointer, believes the
    /// block exists, and writes into nothing (issue #288).
    ///
    /// Stated as a count because the barrier is what the property *is*: the
    /// appends that allocate a block must cost strictly more barriers than the
    /// ones that do not. Both directions are asserted — a run with no allocating
    /// append would satisfy "more" vacuously, and one where every append
    /// allocated would satisfy it without separating anything.
    ///
    /// 320 appends at one chunk each, because a shorter run only ever allocates
    /// *direct* data blocks: the first super block, which has the same hazard
    /// through a different call path, arrives past 300.
    #[test]
    fn allocating_an_index_block_barriers_before_the_pointer_that_names_it() {
        let mut store = WindowProbeStore::open(build_unlimited(4, 1));
        let (mut loc, datatype) = locate(&store);

        // (barriers, blobs appended at end-of-file) for each append.
        let mut rounds: Vec<(usize, usize)> = Vec::new();
        for i in 0..320i32 {
            store.reset_counters();
            append_i32s(&mut store, &mut loc, &datatype, (4 + i)..(5 + i));
            rounds.push((store.syncs.get(), store.appends.get()));
        }

        // A plain append writes only its chunk at end-of-file.
        let base = rounds
            .iter()
            .find(|&&(_, appends)| appends == 1)
            .map(|&(syncs, _)| syncs)
            .expect("some append allocates no index block");
        let blocks = |appends: usize| appends - 1;
        assert!(
            rounds.iter().any(|&(_, appends)| blocks(appends) == 1),
            "the run must allocate a data block somewhere: {rounds:?}"
        );
        assert!(
            rounds.iter().any(|&(_, appends)| blocks(appends) >= 2),
            "the run must reach a super block, which allocates two blocks in one \
             append, or the second barrier site is never exercised: {rounds:?}"
        );
        for (round, &(syncs, appends)) in rounds.iter().enumerate() {
            assert_eq!(
                syncs,
                base + 2 * blocks(appends),
                "append {round} allocated {} index block(s) and took {syncs} \
                 barriers against the {base} for a plain append: each fresh \
                 block requires content and EOF publication barriers",
                blocks(appends)
            );
        }
    }

    #[test]
    fn index_growth_publishes_an_additional_eof_update() {
        let mut store = WindowProbeStore::open(build_unlimited(4, 1));
        let (mut loc, datatype) = locate(&store);

        let mut patches = Vec::new();
        for i in 0..40i32 {
            store.reset_counters();
            append_i32s(&mut store, &mut loc, &datatype, (4 + i)..(5 + i));
            patches.push(store.superblock_patches.get());
        }

        assert!(
            patches.iter().all(|&p| p >= 1),
            "every append extends the file, so every one must record the new \
             end-of-file at least once: {patches:?}"
        );
        assert!(
            patches.contains(&1),
            "an append that allocates no index block must not rewrite the \
             superblock a second time: {patches:?}"
        );
        assert!(
            patches.contains(&2),
            "an append that does allocate one must, or the file would advertise \
             an end-of-file short of its own index: {patches:?}"
        );
        assert!(
            patches.iter().all(|&p| p <= 2),
            "the sequence defines exactly two such points: {patches:?}"
        );
    }

    /// An append whose chunks come out of freed space does not extend the file,
    /// so phase 1 must not rewrite the superblock: the end-of-file it would
    /// record is the one already there (issue #349).
    ///
    /// The reuse itself is the other half of the assertion. Dropping the patch
    /// while still appending would understate the end-of-file on every append —
    /// so this demands the allocation actually came from the free region, and
    /// that the appends resume, with the patch, once that region is spent.
    #[test]
    fn phase_one_patches_the_superblock_only_when_the_file_grew() {
        // A chunk of 4 i32 is 16 bytes; 64 bytes of hole holds four of them.
        let mut store = WindowProbeStore::open(build_unlimited(4, 4)).with_free_tail(64);
        let (mut loc, datatype) = locate(&store);

        let mut rounds = Vec::new();
        for i in 0..6i32 {
            store.reset_counters();
            let base = 4 + i * 4;
            append_i32s(&mut store, &mut loc, &datatype, base..(base + 4));
            rounds.push((
                store.reuses.get(),
                store.appends.get(),
                store.superblock_patches.get(),
            ));
        }

        for &(reuses, appends, patches) in &rounds {
            assert_eq!(
                patches == 0,
                appends == 0,
                "the superblock is rewritten exactly when the file grew: {rounds:?}"
            );
            assert!(reuses + appends >= 1, "every round allocates: {rounds:?}");
        }
        assert!(
            rounds.iter().any(|&(_, appends, _)| appends == 0),
            "the first rounds fit the hole, so they must allocate nothing at \
             end-of-file: {rounds:?}"
        );
        assert!(
            rounds.iter().any(|&(_, appends, _)| appends > 0),
            "the hole is spent well before the last round: {rounds:?}"
        );

        // The reused chunks are the dataset's, not scribble.
        assert_eq!(read_i32s(&store, &loc), (0..28).collect::<Vec<i32>>());
    }

    /// **Every** allocation the append engine makes is served from freed space
    /// when a region fits — the extensible-array blocks that index the chunks as
    /// much as the chunk bytes themselves (issue #349).
    ///
    /// The test store uses one free region for both allocation classes. The fixture's region
    /// is large enough for the chunks and index blocks throughout the append sequence.
    #[test]
    fn a_hole_serves_the_index_blocks_as_well_as_the_chunks() {
        let mut store = WindowProbeStore::open(build_unlimited(4, 1)).with_free_tail(16 * 1024);
        let (mut loc, datatype) = locate(&store);

        let mut rounds = Vec::new();
        for i in 0..40i32 {
            store.reset_counters();
            append_i32s(&mut store, &mut loc, &datatype, (4 + i)..(5 + i));
            rounds.push((store.reuses.get(), store.appends.get()));
        }

        assert!(
            rounds.iter().all(|&(_, appends)| appends == 0),
            "the hole is larger than the whole run, so nothing may reach \
             end-of-file: {rounds:?}"
        );
        assert!(
            rounds.iter().any(|&(reuses, _)| reuses >= 2),
            "some append must allocate an index block beside its chunk, or this \
             says nothing about the blocks: {rounds:?}"
        );
        assert_eq!(
            read_i32s(&store, &loc),
            (0..44).collect::<Vec<i32>>(),
            "the reused regions must hold the dataset's elements"
        );
    }

    /// Relocating a *partial* trailing chunk into freed space is sound: the
    /// element that names it is visible at the old dimension, and phase 2
    /// repoints it to a region whose live prefix reproduces the bytes it already
    /// showed (issue #349).
    ///
    /// This is the one append shape that overwrites an element a reader can
    /// already see, so it is worth exercising against reuse specifically: the
    /// rounds below that grow a partial chunk move it to a fresh address inside
    /// the hole, and the dataset must read back whole every time.
    #[test]
    fn a_relocated_partial_tail_may_land_in_freed_space() {
        const HOLE: u64 = 1024;
        let mut store = WindowProbeStore::open(build_unlimited(4, 4)).with_free_tail(HOLE);
        let hole = (store.len() - HOLE)..store.len();
        let (mut loc, datatype) = locate(&store);

        // (chunks indexed, address of the trailing chunk) after each append.
        let mut tails = Vec::new();
        for i in 4..12i32 {
            append_i32s(&mut store, &mut loc, &datatype, i..(i + 1));
            let tail = loc.num_chunks - 1;
            let addr = loc.read_element(&store, tail).unwrap().unwrap().addr.get();
            tails.push((loc.num_chunks, addr));
            assert_eq!(
                read_i32s(&store, &loc)[..(i as usize + 1)],
                (0..=i).collect::<Vec<i32>>()[..],
                "after appending {i} the live elements must all read back"
            );
        }

        // Relocation means the *same* chunk moved. Comparing consecutive
        // addresses would not say that: the probe allocator never repeats an
        // address, so a run that only ever added fresh chunks would satisfy it.
        assert!(
            tails
                .windows(2)
                .any(|w| w[0].0 == w[1].0 && w[0].1 != w[1].1),
            "a partial trailing chunk must have been rewritten to a new address \
             at least once: {tails:?}"
        );
        assert!(
            tails.iter().all(|&(_, addr)| hole.contains(&addr)),
            "every tail chunk must have landed in the freed region, or this says \
             nothing about reuse: {tails:?}"
        );
    }

    /// The engine's bounded-read contract: appends against a file far larger
    /// than any metadata structure never read more than a bounded window at
    /// once, and never the whole file.
    #[test]
    fn append_reads_stay_bounded_windows() {
        // ~400 KiB of data: far larger than any single bounded window below.
        let n = 100_000i32;
        let mut store = WindowProbeStore::open(build_unlimited(n, 256));
        let file_len = store.data.len();
        assert!(
            file_len > 300_000,
            "test file unexpectedly small: {file_len}"
        );

        let (mut loc, datatype) = locate(&store);
        // Setup (locate + datatype) already obeys the window bound.
        const WINDOW: usize = 16 * 1024;
        assert!(
            store.max_read.get() <= WINDOW,
            "locate read a {}-byte window (> {WINDOW})",
            store.max_read.get()
        );

        // A small append against the large file: every read (trailing chunk,
        // element slots, checksum regions) stays within the window bound, and
        // the total read volume is a small constant, not O(file size).
        store.reset_counters();
        append_i32s(&mut store, &mut loc, &datatype, n..n + 10);
        assert!(
            store.max_read.get() <= WINDOW,
            "append read a {}-byte window (> {WINDOW})",
            store.max_read.get()
        );
        assert!(
            store.total_read.get() <= 64 * 1024,
            "append read {} bytes total (> 64 KiB) on a {file_len}-byte file",
            store.total_read.get()
        );
        assert!(
            store
                .reads
                .borrow()
                .iter()
                .all(|&(_, len)| len < file_len / 2),
            "an append read scaled with file size"
        );

        // Growth loop: the per-append read volume stays flat as the file grows.
        let mut worst = 0usize;
        let mut next = n + 10;
        for _ in 0..20 {
            store.reset_counters();
            append_i32s(&mut store, &mut loc, &datatype, next..next + 300);
            worst = worst.max(store.total_read.get());
            next += 300;
        }
        assert!(
            worst <= 64 * 1024,
            "per-append read volume grew to {worst} bytes"
        );

        // The grown file reads back correctly through the public reader.
        let file = crate::File::from_bytes(store.data).unwrap();
        let ds = file.dataset("d").unwrap();
        let got = ds.read_i32().unwrap();
        let expected: Vec<i32> = (0..next).collect();
        assert_eq!(got.len(), expected.len());
        assert_eq!(got, expected);
    }

    /// Unaligned appends rewrite the trailing partial chunk: still bounded — the
    /// one chunk is the largest data read the plan performs.
    #[test]
    fn partial_tail_append_reads_one_chunk_window() {
        let n = 50_000i32;
        let chunk = 256u64;
        let mut store = WindowProbeStore::open(build_unlimited(n + 7, chunk));
        let (mut loc, datatype) = locate(&store);

        store.reset_counters();
        append_i32s(&mut store, &mut loc, &datatype, n + 7..n + 7 + 13);
        let chunk_bytes = (chunk as usize) * 4;
        assert!(
            store.max_read.get() <= chunk_bytes.max(8 * 1024),
            "partial-tail append read a {}-byte window",
            store.max_read.get()
        );

        let file = crate::File::from_bytes(store.data).unwrap();
        let got = file.dataset("d").unwrap().read_i32().unwrap();
        let expected: Vec<i32> = (0..n + 7 + 13).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn a_filtered_element_width_beyond_the_geometry_limit_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wide_elem.h5");
        let mut b = crate::writer::FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..4096).collect::<Vec<_>>())
            .with_shape(&[4096])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4])
            .with_deflate(DeflateLevel::One);
        b.write(&path).unwrap();

        // The filtered element limit is an 8-byte address, an 8-byte size, and a 4-byte mask.
        let mut bytes = std::fs::read(&path).unwrap();
        let at = bytes
            .windows(4)
            .position(|w| w == b"EAHD")
            .expect("a filtered unlimited dataset is indexed by an extensible array");
        assert_eq!(bytes[at + 6], 14, "the writer's own element width moved");
        bytes[at + 6] = 30;
        crate::checksum::stamp_trailing(&mut bytes, at, 12 + 6 * 8 + 8 + 4);
        std::fs::write(&path, &bytes).unwrap();

        let f = crate::reader::File::open_rw(&path).unwrap();
        let err = f
            .dataset("d")
            .and_then(|mut d| d.append(&[1i32, 2, 3, 4]))
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::Format(FormatError::InvalidExtensibleArrayGeometry)
            ),
            "{err:?}"
        );
        drop(f);
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }

    #[rstest]
    #[case::raw_tail_stays_partial(true, &[106], &[(Stored::Raw, [104, 105, 106, EDGE_FILL])])]
    #[case::raw_tail_becomes_full(true, &[106, 107], &[(Stored::FiltersSkipped, [104, 105, 106, 107])])]
    #[case::raw_tail_becomes_full_before_a_raw_tail(
        true,
        &[106, 107, 108],
        &[
            (Stored::FiltersSkipped, [104, 105, 106, 107]),
            (Stored::Raw, [108, EDGE_FILL, EDGE_FILL, EDGE_FILL]),
        ],
    )]
    #[case::filtered_tail_stays_partial(
        false,
        &[106],
        &[(Stored::Filtered, [104, 105, 106, EDGE_FILL])],
    )]
    #[case::filtered_tail_becomes_full(
        false,
        &[106, 107],
        &[(Stored::Filtered, [104, 105, 106, 107])],
    )]
    #[case::filtered_tail_becomes_full_before_a_filtered_tail(
        false,
        &[106, 107, 108],
        &[
            (Stored::Filtered, [104, 105, 106, 107]),
            (Stored::Filtered, [108, EDGE_FILL, EDGE_FILL, EDGE_FILL]),
        ],
    )]
    fn every_publication_state_of_an_edge_append_reads_under_its_extent(
        #[case] raw_edges: bool,
        #[case] appended: &[i32],
        #[case] written: &[(Stored, [i32; 4])],
        #[values(EdgeFilter::Deflate, EdgeFilter::Shuffle, EdgeFilter::ShuffleDeflate)]
        filter: EdgeFilter,
        #[values(1, 2, 3, 4)] max_phase: u8,
    ) {
        let mut store = edge_store(filter, raw_edges);
        let before = stored_chunks(&store.data);
        append_to_edge_store(&mut store, appended, max_phase).unwrap();

        let written = filter.stored_chunks_after_the_kept_one(written);
        let (chunks, values) = match max_phase {
            1 => (before, EDGE_INITIAL.to_vec()),
            2 | 3 => (
                [&before[..1], &written[..1]].concat(),
                EDGE_INITIAL.to_vec(),
            ),
            _ => (
                [&before[..1], &written].concat(),
                [&EDGE_INITIAL[..], appended].concat(),
            ),
        };
        assert_eq!(stored_chunks(&store.data), chunks);
        let file = crate::File::from_bytes(store.data).unwrap();
        assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), values);
    }

    #[rstest]
    #[case::payload_barrier(4, Some(1), &[1.46, 1.56], 0)]
    #[case::eof_barrier(4, Some(2), &[1.46, 1.56], 0)]
    #[case::index_barrier(4, Some(3), &[1.46, 1.56], 1)]
    #[case::header_barrier(4, Some(4), &[1.46, 1.56], 1)]
    #[case::payloads(1, None, &[1.46, 1.56], 0)]
    #[case::index(2, None, &[1.46, 1.56], 1)]
    #[case::header(3, None, &[1.46, 1.56], 1)]
    #[case::extent_barrier(4, Some(5), &[1.46, 1.56, 1.0, 2.0], 1)]
    #[case::filtered_barrier(4, Some(6), &[1.5, 1.6, 1.0, 2.0], 0)]
    #[case::complete(4, None, &[1.5, 1.6, 1.0, 2.0], 0)]
    fn lossy_raw_tail_publication_remains_readable_at_each_barrier(
        #[case] phase: u8,
        #[case] fail_sync: Option<usize>,
        #[case] expected: &[f64],
        #[case] mask: u32,
    ) {
        let mut builder = FileBuilder::new();
        builder
            .create_dataset("d")
            .with_f64_data(&[1.46, 1.56])
            .with_shape(&[2])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_create_properties(
                DatasetCreateProperties::new()
                    .with_chunks(&[4])
                    .with_scale_offset(ScaleOffset::FloatDScale(1))
                    .with_partial_edge_filtering(PartialEdgeFiltering::Skip),
            );
        let mut store = WindowProbeStore::open(builder.finish().unwrap());
        let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
        let original = state
            .index
            .allocated()
            .unwrap()
            .read_element(&store, 0)
            .unwrap()
            .unwrap();
        let original = FileRange::new(
            store.addresses().resolve(original.addr).unwrap(),
            original.stored_size,
        )
        .unwrap();
        let intermediate = FileRange::new(PhysicalAddress::new(store.len()), 32).unwrap();
        let raw: Vec<_> = [1.0_f64, 2.0]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect();
        let plan = plan_ea_append(
            &store,
            state.index.allocated().unwrap(),
            &state.datatype,
            &state.spatial,
            state.element_size,
            state.pipeline.as_ref(),
            true,
            LossyTailReencoding::Reject,
            &raw,
            2,
            state.fill.pattern(state.element_size),
        )
        .unwrap();
        store.fail_sync = fail_sync;
        match apply_ea_append(
            &mut store,
            state.index.allocated_mut().unwrap(),
            plan,
            phase,
        ) {
            Ok(()) => assert_eq!(fail_sync, None),
            Err(Error::Io(error)) => {
                assert!(fail_sync.is_some());
                assert_eq!(error.kind(), std::io::ErrorKind::Other);
                assert_eq!(error.to_string(), "injected synchronization failure");
            }
            Err(error) => panic!("expected a synchronization error, got {error:?}"),
        }
        let mut retired = Vec::new();
        if phase >= 2 && fail_sync.is_none_or(|barrier| barrier > 3) {
            retired.push(original);
        }
        if phase == 4 && fail_sync.is_none() {
            retired.push(intermediate);
        }
        assert_eq!(store.retired, retired);
        let file = crate::File::from_bytes(store.data).unwrap();
        let dataset = file.dataset("d").unwrap();
        assert_eq!(dataset.read_f64().unwrap(), expected);
        let chunks = dataset.chunks().unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].filter_mask().raw_bits(), mask);
    }

    #[rstest]
    #[case::other_dataset(
        ReferenceChange::UseOtherDataset,
        "append plan refers to a different chunk index"
    )]
    #[case::duplicate_before_extent(
        ReferenceChange::PublishIndex,
        "chunk reference changed after append planning"
    )]
    #[case::duplicate_after_extent(
        ReferenceChange::PublishExtent,
        "chunk reference changed after append planning"
    )]
    fn captured_tail_references_reject_unrelated_or_duplicate_publication(
        #[case] change: ReferenceChange,
        #[case] expected: &str,
    ) {
        let mut builder = FileBuilder::new();
        for name in ["d", "e"] {
            builder
                .create_dataset(name)
                .with_f64_data(&[1.46, 1.56])
                .with_maxshape(&[MaxExtent::Unlimited])
                .with_create_properties(
                    DatasetCreateProperties::new()
                        .with_chunks(&[4])
                        .with_scale_offset(ScaleOffset::FloatDScale(1))
                        .with_partial_edge_filtering(PartialEdgeFiltering::Skip),
                );
        }
        let mut store = WindowProbeStore::open(builder.finish().unwrap());
        let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
        let plan = lossy_completion_plan(&store, &state);
        let confirmed;
        let error = match change {
            ReferenceChange::PublishExtent | ReferenceChange::PublishIndex => {
                let first = lossy_completion_plan(&store, &state);
                let phase = if matches!(change, ReferenceChange::PublishIndex) {
                    2
                } else {
                    4
                };
                apply_ea_append(
                    &mut store,
                    state.index.allocated_mut().unwrap(),
                    first,
                    phase,
                )
                .unwrap();
                confirmed = store.retired.clone();
                assert_eq!(confirmed.len(), if phase == 2 { 1 } else { 2 });
                apply_ea_append(&mut store, state.index.allocated_mut().unwrap(), plan, 4)
                    .unwrap_err()
            }
            ReferenceChange::UseOtherDataset => {
                let address = group_v2::resolve_path_any(
                    PhysicalBytes::for_test(&store.data, store.addresses()),
                    AccessMode::ReadOnly,
                    &store.superblock,
                    store.widths,
                    &ObjectPath::parse("e"),
                )
                .unwrap();
                let mut other = edit::locate_dataset_state(
                    &store,
                    ObjectId::new(PhysicalAddress::new(address)),
                )
                .unwrap();
                confirmed = Vec::new();
                apply_ea_append(&mut store, other.index.allocated_mut().unwrap(), plan, 4)
                    .unwrap_err()
            }
        };
        let Error::AppendUnsupported(reason) = error else {
            panic!("expected a changed chunk reference, got {error:?}");
        };
        assert_eq!(reason, expected);
        assert_eq!(store.retired, confirmed);
        let file = crate::File::from_bytes(store.data).unwrap();
        assert_eq!(file.dataset("e").unwrap().read_f64().unwrap(), [1.46, 1.56]);
        assert_eq!(
            file.dataset("d").unwrap().read_f64().unwrap(),
            match change {
                ReferenceChange::PublishExtent => vec![1.5, 1.6, 1.0, 2.0],
                ReferenceChange::PublishIndex | ReferenceChange::UseOtherDataset =>
                    vec![1.46, 1.56],
            }
        );
    }

    #[derive(Clone, Copy)]
    enum ReferenceChange {
        PublishExtent,
        PublishIndex,
        UseOtherDataset,
    }

    fn lossy_completion_plan(store: &WindowProbeStore, state: &LocatedState) -> AppendPlan {
        plan_ea_append(
            store,
            state.index.allocated().unwrap(),
            &state.datatype,
            &state.spatial,
            state.element_size,
            state.pipeline.as_ref(),
            true,
            LossyTailReencoding::Reject,
            &[1.0_f64, 2.0]
                .into_iter()
                .flat_map(f64::to_le_bytes)
                .collect::<Vec<_>>(),
            2,
            state.fill.pattern(state.element_size),
        )
        .unwrap()
    }

    #[rstest]
    #[case::partial_payloads(1, None, &[-0.26], &[1.46, 1.56])]
    #[case::partial_index(2, None, &[-0.26], &[1.44, 1.54])]
    #[case::partial_header(3, None, &[-0.26], &[1.44, 1.54])]
    #[case::partial_index_barrier(4, Some(3), &[-0.26], &[1.44, 1.54])]
    #[case::partial_extent_barrier(4, Some(5), &[-0.26], &[1.44, 1.54, -0.26])]
    #[case::partial_complete(4, None, &[-0.26], &[1.44, 1.54, -0.26])]
    #[case::full_payloads(1, None, &[-0.26, 2.0], &[1.46, 1.56])]
    #[case::full_index(2, None, &[-0.26, 2.0], &[1.44, 1.54])]
    #[case::full_header(3, None, &[-0.26, 2.0], &[1.44, 1.54])]
    #[case::full_index_barrier(4, Some(3), &[-0.26, 2.0], &[1.44, 1.54])]
    #[case::full_extent_barrier(4, Some(5), &[-0.26, 2.0], &[1.44, 1.54, -0.26, 2.04])]
    #[case::full_complete(4, None, &[-0.26, 2.0], &[1.44, 1.54, -0.26, 2.04])]
    fn lossy_tail_reencoding_keeps_each_recorded_extent_readable(
        #[case] phase: u8,
        #[case] fail_sync: Option<usize>,
        #[case] appended: &[f64],
        #[case] expected: &[f64],
    ) {
        let mut builder = FileBuilder::new();
        builder
            .create_dataset("d")
            .with_f64_data(&[1.46, 1.56])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4])
            .with_scale_offset(ScaleOffset::FloatDScale(1));
        let mut store = WindowProbeStore::open(builder.finish().unwrap());
        let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
        let original = state
            .index
            .allocated()
            .unwrap()
            .read_element(&store, 0)
            .unwrap()
            .unwrap();
        let original = FileRange::new(
            store.addresses().resolve(original.addr).unwrap(),
            original.stored_size,
        )
        .unwrap();
        let plan = plan_ea_append(
            &store,
            state.index.allocated().unwrap(),
            &state.datatype,
            &state.spatial,
            state.element_size,
            state.pipeline.as_ref(),
            true,
            LossyTailReencoding::Allow,
            &appended
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
            appended.len().to_u64(),
            state.fill.pattern(state.element_size),
        )
        .unwrap();
        store.fail_sync = fail_sync;
        match apply_ea_append(
            &mut store,
            state.index.allocated_mut().unwrap(),
            plan,
            phase,
        ) {
            Ok(()) => assert_eq!(fail_sync, None),
            Err(Error::Io(error)) => {
                assert!(fail_sync.is_some());
                assert_eq!(error.kind(), std::io::ErrorKind::Other);
                assert_eq!(error.to_string(), "injected synchronization failure");
            }
            Err(error) => panic!("expected a synchronization error, got {error:?}"),
        }
        let expected_retired = if phase >= 2 && fail_sync.is_none_or(|barrier| barrier > 3) {
            vec![original]
        } else {
            Vec::new()
        };
        assert_eq!(store.retired, expected_retired);
        let file = crate::File::from_bytes(store.data).unwrap();
        let dataset = file.dataset("d").unwrap();
        assert_eq!(dataset.read_f64().unwrap(), expected);
        let chunks = dataset.chunks().unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].filter_mask().raw_bits(), 0);
    }

    #[test]
    fn filtered_retirement_tracks_growing_and_shrinking_stored_lengths() {
        let mut builder = FileBuilder::new();
        builder
            .create_dataset("d")
            .with_i32_data(&[0])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[512])
            .with_deflate(DeflateLevel::Six);
        let mut store = WindowProbeStore::open(builder.finish().unwrap());
        let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
        let mut expected = vec![0_i32];
        let mut grows = 0;
        let mut shrinks = 0;
        for value in 1_i32..400 {
            let value = value % 5;
            let predecessor = state
                .index
                .allocated()
                .unwrap()
                .read_element(&store, 0)
                .unwrap()
                .unwrap();
            let range = FileRange::new(
                store.addresses().resolve(predecessor.addr).unwrap(),
                predecessor.stored_size,
            )
            .unwrap();
            let plan = plan_ea_append(
                &store,
                state.index.allocated().unwrap(),
                &state.datatype,
                &state.spatial,
                state.element_size,
                state.pipeline.as_ref(),
                true,
                LossyTailReencoding::Reject,
                &value.to_le_bytes(),
                1,
                state.fill.pattern(state.element_size),
            )
            .unwrap();
            apply_ea_append(&mut store, state.index.allocated_mut().unwrap(), plan, 4).unwrap();
            assert_eq!(store.retired.pop(), Some(range));
            assert!(store.retired.is_empty());
            let replacement = state
                .index
                .allocated()
                .unwrap()
                .read_element(&store, 0)
                .unwrap()
                .unwrap();
            grows += usize::from(replacement.stored_size > predecessor.stored_size);
            shrinks += usize::from(replacement.stored_size < predecessor.stored_size);
            expected.push(value);
        }
        assert!(grows > 0, "no encoded allocation grew");
        assert!(shrinks > 0, "no encoded allocation shrank");
        let file = crate::File::from_bytes(store.data).unwrap();
        assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), expected);
    }

    #[rstest]
    #[case::short_raw_tail(true, 12)]
    #[case::long_raw_tail(true, 20)]
    #[case::short_filtered_tail(false, 12)]
    #[case::long_filtered_tail(false, 20)]
    fn an_old_tail_of_another_decoded_length_is_rejected(
        #[case] raw_edges: bool,
        #[case] decoded_len: usize,
    ) {
        let store = edge_store_decoding_to(raw_edges, decoded_len);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edge.h5");
        std::fs::write(&path, &store.data).unwrap();

        let file = crate::File::open_rw(&path).unwrap();
        let err = file.dataset("d").unwrap().append(&[106_i32]).unwrap_err();
        let Error::Format(FormatError::DataSizeMismatch { expected, actual }) = err else {
            panic!("expected DataSizeMismatch, got {err:?}");
        };
        assert_eq!((expected, actual), (16, decoded_len));
        file.close().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), store.data);
    }

    #[rstest]
    #[case::raw_tail_stays_partial(true, &[106], &[(Stored::Raw, [104, 105, 106, EDGE_FILL])])]
    #[case::raw_tail_becomes_full(true, &[106, 107], &[(Stored::Filtered, [104, 105, 106, 107])])]
    #[case::raw_tail_becomes_full_before_a_raw_tail(
        true,
        &[106, 107, 108],
        &[
            (Stored::Filtered, [104, 105, 106, 107]),
            (Stored::Raw, [108, EDGE_FILL, EDGE_FILL, EDGE_FILL]),
        ],
    )]
    #[case::filtered_tail_stays_partial(
        false,
        &[106],
        &[(Stored::Filtered, [104, 105, 106, EDGE_FILL])],
    )]
    #[case::filtered_tail_becomes_full(
        false,
        &[106, 107],
        &[(Stored::Filtered, [104, 105, 106, 107])],
    )]
    #[case::filtered_tail_becomes_full_before_a_filtered_tail(
        false,
        &[106, 107, 108],
        &[
            (Stored::Filtered, [104, 105, 106, 107]),
            (Stored::Filtered, [108, EDGE_FILL, EDGE_FILL, EDGE_FILL]),
        ],
    )]
    fn a_staged_edge_append_keeps_the_layout_flags_and_stores_each_written_chunk_for_the_new_extent(
        #[case] raw_edges: bool,
        #[case] appended: &[i32],
        #[case] written: &[(Stored, [i32; 4])],
        #[values(EdgeFilter::Deflate, EdgeFilter::Shuffle, EdgeFilter::ShuffleDeflate)]
        filter: EdgeFilter,
    ) {
        let store = edge_store(filter, raw_edges);
        let before = stored_chunks(&store.data);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edge.h5");
        std::fs::write(&path, &store.data).unwrap();

        let file = crate::File::open_rw(&path).unwrap();
        let mut dataset = file.dataset("d").unwrap();
        dataset
            .append_staged(|builder| {
                builder.append(appended);
            })
            .unwrap();
        drop(dataset);
        file.close().unwrap();

        let data = std::fs::read(&path).unwrap();
        assert_eq!(
            stored_chunks(&data),
            [
                &before[..1],
                &filter.stored_chunks_after_the_kept_one(written)
            ]
            .concat()
        );
        let flags = if raw_edges {
            crate::data_layout::DONT_FILTER_PARTIAL_BOUND_CHUNKS
        } else {
            0
        };
        assert_eq!(
            locate(&WindowProbeStore::open(data.clone())).0.flags,
            ChunkedLayoutFlags::new(flags)
        );
        let file = crate::File::from_bytes(data).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_i32().unwrap(),
            [&EDGE_INITIAL[..], appended].concat()
        );
    }

    #[rstest]
    #[case::short_raw_tail(true, 12)]
    #[case::long_raw_tail(true, 20)]
    #[case::short_filtered_tail(false, 12)]
    #[case::long_filtered_tail(false, 20)]
    fn a_staged_append_onto_an_old_tail_of_another_decoded_length_is_rejected(
        #[case] raw_edges: bool,
        #[case] decoded_len: usize,
    ) {
        let store = edge_store_decoding_to(raw_edges, decoded_len);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edge.h5");
        std::fs::write(&path, &store.data).unwrap();

        let file = crate::File::open_rw(&path).unwrap();
        let mut dataset = file.dataset("d").unwrap();
        dataset
            .append_staged(|builder| {
                builder.append(&[106_i32]);
            })
            .unwrap();
        let err = file.commit().unwrap_err();
        let Error::Format(FormatError::DataSizeMismatch { expected, actual }) = err else {
            panic!("expected DataSizeMismatch, got {err:?}");
        };
        assert_eq!((expected, actual), (16, decoded_len));
        drop(dataset);
        drop(file);
        assert_eq!(std::fs::read(&path).unwrap(), store.data);
    }

    #[derive(Clone, Copy, Debug)]
    enum EdgeFilter {
        Deflate,
        Shuffle,
        ShuffleDeflate,
    }

    impl EdgeFilter {
        fn pipeline(self) -> FilterPipeline {
            let store = self.store();
            edit::locate_dataset_state(&store, dataset_address(&store))
                .unwrap()
                .pipeline
                .unwrap()
        }

        fn every_filter_skipped(self) -> u32 {
            match self {
                Self::Deflate | Self::Shuffle => 0b1,
                Self::ShuffleDeflate => 0b11,
            }
        }

        fn stored_chunks_after_the_kept_one(
            self,
            written: &[(Stored, [i32; 4])],
        ) -> Vec<(Vec<u64>, u32, Vec<u8>)> {
            let pipeline = self.pipeline();
            (4..)
                .step_by(4)
                .zip(written)
                .map(|(offset, (stored, values))| {
                    let logical = i32_bytes(values);
                    let (filter_mask, bytes) = match stored {
                        Stored::Filtered => (
                            0,
                            filters::compress_chunk(
                                &logical,
                                &pipeline,
                                ChunkContext::basic(&[4], 4),
                            )
                            .unwrap(),
                        ),
                        Stored::FiltersSkipped => (self.every_filter_skipped(), logical),
                        Stored::Raw => (0, logical),
                    };
                    (vec![offset], filter_mask, bytes)
                })
                .collect()
        }

        fn store(self) -> WindowProbeStore {
            let mut builder = FileBuilder::new();
            let dataset = builder
                .create_dataset("d")
                .with_i32_data(&EDGE_INITIAL)
                .with_shape(&[6])
                .with_maxshape(&[MaxExtent::Unlimited])
                .with_chunks(&[4])
                .with_fill_value(EDGE_FILL);
            match self {
                Self::Deflate => dataset.with_deflate(DeflateLevel::Six),
                Self::Shuffle => dataset.with_shuffle(),
                Self::ShuffleDeflate => dataset.with_shuffle().with_deflate(DeflateLevel::Six),
            };
            WindowProbeStore::open(builder.finish().unwrap())
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Stored {
        Filtered,
        FiltersSkipped,
        Raw,
    }

    fn edge_store(filter: EdgeFilter, raw_edges: bool) -> WindowProbeStore {
        let mut store = filter.store();
        if raw_edges {
            let walk = walk_v2_object_header(
                &store,
                dataset_address(&store).header_address(),
                store.widths,
            )
            .unwrap();
            let layout = walk
                .messages
                .iter()
                .find(|m| m.msg_type == MessageType::DATA_LAYOUT)
                .unwrap();
            assert_eq!(
                store.data[layout.data_off.to_usize().unwrap()],
                LayoutVersion::Four.get()
            );
            store
                .publish_checksummed(
                    layout.chunk_start,
                    layout.chunk_msg_end,
                    layout.data_off + LAYOUT_PROPERTIES_OFFSET.to_u64(),
                    &[crate::data_layout::DONT_FILTER_PARTIAL_BOUND_CHUNKS],
                )
                .unwrap();
            replace_edge_record(&mut store, &i32_bytes(&[104, 105, EDGE_FILL, EDGE_FILL]));
        }
        store
    }

    fn edge_store_decoding_to(raw_edges: bool, decoded_len: usize) -> WindowProbeStore {
        let mut store = edge_store(EdgeFilter::Shuffle, raw_edges);
        let logical = vec![0x5a; decoded_len];
        let stored = if raw_edges {
            logical
        } else {
            filters::compress_chunk(
                &logical,
                &EdgeFilter::Shuffle.pipeline(),
                ChunkContext::basic(&[4], 4),
            )
            .unwrap()
        };
        replace_edge_record(&mut store, &stored);
        store
    }

    fn replace_edge_record(store: &mut WindowProbeStore, stored: &[u8]) {
        let (mut loc, _) = locate(store);
        let allocation = store.allocate(stored, AllocationClass::Raw).unwrap();
        let publication = store.publish_allocations().unwrap();
        let rec = ConfirmedChunk {
            allocation: publication.confirm(allocation).unwrap(),
            filter_mask: 0,
        };
        loc.ea_insert(store, 1, rec).unwrap();
    }

    fn append_to_edge_store(
        store: &mut WindowProbeStore,
        values: &[i32],
        max_phase: u8,
    ) -> Result<(), Error> {
        let mut state = edit::locate_dataset_state(&*store, dataset_address(store)).unwrap();
        let plan = plan_ea_append(
            store,
            state.index.allocated().unwrap(),
            &state.datatype,
            &state.spatial,
            state.element_size,
            state.pipeline.as_ref(),
            true,
            LossyTailReencoding::Reject,
            &i32_bytes(values),
            values.len() as u64,
            state.fill.pattern(state.element_size),
        )?;
        apply_ea_append(store, state.index.allocated_mut().unwrap(), plan, max_phase)
    }

    fn stored_chunks(data: &[u8]) -> Vec<(Vec<u64>, u32, Vec<u8>)> {
        crate::File::from_bytes(data.to_vec())
            .unwrap()
            .dataset("d")
            .unwrap()
            .chunks()
            .unwrap()
            .into_iter()
            .map(|chunk| {
                let at = chunk.address.to_usize().unwrap();
                let end = at + chunk.storage_size.to_usize().unwrap();
                let filter_mask = chunk.filter_mask().raw_bits();
                (chunk.offset, filter_mask, data[at..end].to_vec())
            })
            .collect()
    }

    fn i32_bytes(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    const EDGE_INITIAL: [i32; 6] = [100, 101, 102, 103, 104, 105];
    const EDGE_FILL: i32 = -7;

    // "The Extensible Array Index" in format specification, version 4.0, defines the version 0
    // header element-size field and the four-byte filter mask for version 4 chunked layouts.
    const EA_HEADER_ELEMENT_SIZE: usize = 6;
    const EA_FILTER_MASK_BYTES: u8 = 4;

    // "The Extensible Array Index" in format specification, version 4.0, defines the version 0
    // header signature and version.
    const EA_HEADER_SIGNATURE: &[u8; 4] = b"EAHD";
    const EA_HEADER_VERSION: u8 = 0;

    // "The Extensible Array Index" in format specification, version 4.0, places six statistics
    // after the version 0 header's byte fields.
    const EA_STATS_OFFSET: u64 = 12;
    const EA_STAT_FIELDS: usize = 6;
    // "The Data Layout Message" in the format specification, version 4.0, defines the version 4
    // chunked-layout prefix: flags, dimensionality, and dimension-size encoded length.
    const V4_DIMENSION_COUNT_OFFSET: usize = LAYOUT_PROPERTIES_OFFSET + 1;
    const V4_DIMENSION_WIDTH_OFFSET: usize = V4_DIMENSION_COUNT_OFFSET + 1;
    const V4_DIMENSIONS_OFFSET: usize = V4_DIMENSION_WIDTH_OFFSET + 1;
    const FIXTURE_DIMENSION_BYTES: u8 = 8;
}
