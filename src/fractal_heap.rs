//! Resolves managed, huge, and tiny objects stored in a fractal heap.
//!
//! [`HeapObjectReader`] reads from either an in-memory file image or a [`Source`].

use alloc::borrow::Cow;
use alloc::collections::BTreeSet;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT;
use hdf5_pure_format::__private::FractalHeapBlock;
use hdf5_pure_format::__private::FractalHeapBlockLayout;
pub use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::FractalHeapHeaderLocator;
use hdf5_pure_format::__private::FractalHeapIdKind;
use hdf5_pure_format::__private::FractalHeapIdView;
use hdf5_pure_format::__private::HugeObjectRecord;
use hdf5_pure_format::__private::HugeObjectReference;

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::btree_v2::{
    BTreeV2Header, BTreeV2Record, collect_btree_v2_records, collect_btree_v2_records_from_source,
};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::Source;
use crate::source::SourceMetadata;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// Parses a heap header from an image framed at the effective base in `addresses`.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the image is truncated, and the address, width,
/// and header parsing errors of [`read_header_from_source`].
pub(crate) fn parse_header(
    bytes: &[u8],
    addresses: AddressContext,
    address: StoredAddress,
    offset_size: u8,
    length_size: u8,
) -> Result<FractalHeapHeader, FormatError> {
    let image = FileRegion::new(
        FileRange::new(
            addresses.resolve(StoredAddress::new(0))?,
            bytes.len() as u64,
        )?,
        bytes,
    )?;
    let locator = FractalHeapHeaderLocator::new(
        addresses.resolve(address)?,
        OffsetWidth::try_from(offset_size)?,
        LengthWidth::try_from(length_size)?,
    );
    let prefix = buffered_region(image, locator.parse_range()?)?;
    let reference = locator.parse(prefix)?;
    reference.parse(buffered_region(image, reference.parse_range())?)
}

fn buffered_region<'a>(
    image: FileRegion<'a>,
    range: FileRange,
) -> Result<FileRegion<'a>, FormatError> {
    if !image.range().contains(range) {
        return Err(FormatError::UnexpectedEof {
            expected: range
                .end()
                .get()
                .saturating_sub(image.range().start().get())
                .to_usize()
                .unwrap_or(usize::MAX),
            available: image.bytes().len(),
        });
    }
    image.subregion(range)
}

/// Reads a fractal heap header using its prefix and complete header ranges.
///
/// `source` is framed at the effective base in `addresses`. The parser receives physical
/// [`FileRegion`] coordinates, and each source read uses a base-relative offset.
///
/// # Errors
///
/// Returns [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a field
/// width is not 2, 4, or 8, [`FormatError::OffsetOverflow`] if an address or range end exceeds
/// [`u64::MAX`], and the parsing errors of [`FractalHeapHeaderLocator::parse`] and
/// [`FractalHeapHeaderRef::parse`](hdf5_pure_format::__private::FractalHeapHeaderRef::parse).
/// Returns the source error if a read fails.
pub(crate) fn read_header_from_source(
    source: &(impl Source + ?Sized),
    addresses: AddressContext,
    address: StoredAddress,
    offset_size: u8,
    length_size: u8,
) -> Result<FractalHeapHeader, FormatError> {
    let locator = FractalHeapHeaderLocator::new(
        addresses.resolve(address)?,
        OffsetWidth::try_from(offset_size)?,
        LengthWidth::try_from(length_size)?,
    );
    let prefix_range = locator.parse_range()?;
    let prefix = read_header_range(source, addresses, prefix_range)?;
    let header_ref = locator.parse(FileRegion::new(prefix_range, &prefix)?)?;
    let range = header_ref.parse_range();
    let bytes = read_header_range(source, addresses, range)?;
    header_ref.parse(FileRegion::new(range, &bytes)?)
}

/// Reads a physical header range through a source framed at `addresses`' effective base.
///
/// # Errors
///
/// Returns [`FormatError::AddressBelowBase`] if the range starts below the effective base,
/// [`FormatError::ValueTooLargeForPlatform`] if the read length exceeds [`usize::MAX`], and the
/// source error if a read fails.
pub(crate) fn read_header_range(
    source: &(impl Source + ?Sized),
    addresses: AddressContext,
    range: FileRange,
) -> Result<Vec<u8>, FormatError> {
    source.read_metadata_at(
        addresses.relative(range.start().get())?.get(),
        range.length().to_usize()?,
    )
}

/// Copy `len` bytes at `addr` out of an in-memory file image, bounds-checked.
fn slice_object(file_data: &[u8], addr: u64, len: usize) -> Result<Vec<u8>, FormatError> {
    let start = addr.to_usize()?;
    let end = start.checked_add(len).ok_or(FormatError::OffsetOverflow {
        offset: addr,
        length: len as u64,
    })?;
    if end > file_data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: end,
            available: file_data.len(),
        });
    }
    Ok(file_data[start..end].to_vec())
}

/// Read `len` bytes at `addr` from a streaming source.
fn read_object_at_source<S: Source + ?Sized>(
    source: &S,
    addr: u64,
    len: usize,
) -> Result<Vec<u8>, FormatError> {
    source.read_metadata_at(addr, len)
}

/// Checks the B-tree header against the indexed, unfiltered huge-object record layout.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedHugeObjectBTree`] if the record type differs or the
/// records are too short, and [`FormatError::InvalidOffsetSize`] or
/// [`FormatError::InvalidLengthSize`] if a field width is not 2, 4, or 8.
fn check_huge_object_btree(
    header: &BTreeV2Header,
    offset_size: u8,
    length_size: u8,
) -> Result<(), FormatError> {
    let required = HugeObjectRecord::size(
        OffsetWidth::try_from(offset_size)?,
        LengthWidth::try_from(length_size)?,
    );
    if header.tree_type != BTREE_V2_HUGE_OBJECT || header.record_size < required {
        return Err(FormatError::UnexpectedHugeObjectBTree {
            tree_type: header.tree_type,
            record_size: usize::from(header.record_size),
            required: usize::from(required),
        });
    }
    Ok(())
}

/// Every huge object a heap holds, decoded from its huge-objects v2 B-tree.
///
/// Resolving even one huge object costs the whole index: that B-tree is read by
/// collecting its records rather than by descending to a key. So the records are
/// decoded here once, and kept by the [`HeapObjectReader`] that asked for them.
///
/// Only the type-1 record layout appears here, which
/// [`check_huge_object_btree`] establishes before any record reaches this.
struct HugeObjectIndex {
    /// `(id, address, length)`, ordered by id so a lookup is a binary search.
    /// The B-tree already stores them in that order; sorting again costs one
    /// pass and means correctness here does not rest on that.
    entries: Vec<(u64, StoredAddress, u64)>,
}

#[cfg(test)]
std::thread_local! {
    /// How many huge-object indexes have been decoded on this thread.
    ///
    /// Parsing one index per object rather than one per walk returns exactly the
    /// same objects while making the walk quadratic in their number, so the
    /// count of decodes is the only thing a test can hold the invariant to. The
    /// walks build their reader internally, so the count has to be reachable
    /// from outside the reader that did the decoding; it is per-thread because
    /// the test harness runs tests concurrently, and each test's walk runs on
    /// its own thread.
    static HUGE_INDEX_DECODES: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// Start counting huge-object index decodes on this thread from zero.
#[cfg(test)]
pub(crate) fn reset_huge_index_decodes() {
    HUGE_INDEX_DECODES.with(|count| count.set(0));
}

/// How many huge-object indexes this thread has decoded since the last reset.
#[cfg(test)]
pub(crate) fn huge_index_decodes() -> usize {
    HUGE_INDEX_DECODES.with(|count| count.get())
}

impl HugeObjectIndex {
    fn decode(
        records: &[BTreeV2Record],
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        #[cfg(test)]
        HUGE_INDEX_DECODES.with(|count| count.set(count.get() + 1));

        let offset_width = OffsetWidth::try_from(offset_size)?;
        let length_width = LengthWidth::try_from(length_size)?;
        let mut entries = Vec::with_capacity(records.len());
        for record in records {
            let HugeObjectRecord {
                address,
                length,
                id,
            } = record.huge_object(offset_width, length_width)?;
            entries.push((id, address, length));
        }
        entries.sort_unstable();
        Ok(Self { entries })
    }

    /// The `(address, length)` of the object with `huge_id`.
    fn locate(&self, huge_id: u64) -> Result<(StoredAddress, u64), FormatError> {
        match self
            .entries
            .binary_search_by_key(&huge_id, |&(id, _, _)| id)
        {
            Ok(i) => {
                let (_, addr, len) = self.entries[i];
                Ok((addr, len))
            }
            Err(_) => Err(FormatError::HugeObjectNotFound(huge_id)),
        }
    }
}

/// Reads objects from one fractal heap using parsed heap IDs.
///
/// The caller parses each [`FractalHeapIdView`] using the decoding parameters of the heap passed
/// to [`new`](Self::new). The reader caches the huge-object B-tree records after the first indexed
/// huge-object lookup. Inline huge objects and tiny objects are read directly from their parsed
/// fields.
///
/// For read-only managed-object lookups, the reader validates the direct block's prefix and
/// object bounds and reads only the object bytes. With read-write access, the reader
/// validates a checksummed direct block in full before reading an object and caches successful
/// validation for subsequent objects in that block. Validation caches require an unchanged image
/// and one backend for the lifetime of the reader.
pub struct HeapObjectReader<'h> {
    header: &'h FractalHeapHeader,
    offset_size: u8,
    length_size: u8,
    addresses: AddressContext,
    access_mode: AccessMode,
    /// Direct blocks whose complete checksums were validated against this reader's image.
    validated_direct: BTreeSet<StoredAddress>,
    validated_backend: Option<Backend>,
    layout: Result<FractalHeapBlockLayout, FormatError>,
    /// Parsed on demand, since most heaps hold no huge object at all.
    huge: Option<HugeObjectIndex>,
    /// Which backend the cached index was decoded from, once one has been.
    huge_backend: Option<Backend>,
}

/// Which of the two ways of reading a file an object came through. Recorded
/// only to hold a reader to one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    /// An in-memory image of the whole file.
    Buffered,
    /// A [`Source`], read from on demand.
    Streaming,
}

impl<'h> HeapObjectReader<'h> {
    /// Returns a reader of the objects of the heap `header` describes.
    ///
    /// `owner` is the header's stored address. `addresses` resolves addresses against the image's
    /// effective base. The image or source must be framed at that base and stay unchanged for
    /// the reader's lifetime. `access_mode` selects bounded read-only lookup or mandatory
    /// read-write checksum validation. The widths are the superblock's "Size of Offsets" and
    /// "Size of Lengths" bytes.
    pub(crate) fn new(
        header: &'h FractalHeapHeader,
        addresses: AddressContext,
        owner: StoredAddress,
        access_mode: AccessMode,
        offset_size: u8,
        length_size: u8,
    ) -> Self {
        Self {
            header,
            addresses,
            access_mode,
            validated_direct: BTreeSet::new(),
            validated_backend: None,
            offset_size,
            length_size,
            layout: if header.filtering.is_filtered() {
                Err(FormatError::UnsupportedFilteredHeapObject)
            } else {
                OffsetWidth::try_from(offset_size).and_then(|offsets| {
                    LengthWidth::try_from(length_size).and_then(|lengths| {
                        FractalHeapBlockLayout::new(header, offsets, lengths, owner)
                    })
                })
            },
            huge: None,
            huge_backend: None,
        }
    }

    fn managed_layout(&self) -> Result<&FractalHeapBlockLayout, FormatError> {
        self.layout.as_ref().map_err(Clone::clone)
    }

    /// Reads the object identified by a parsed ID from a file image.
    ///
    /// Managed objects are read through the heap's doubling table. Indexed huge objects are
    /// looked up in the cached huge-object index. Tiny-object bytes are copied from `id`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFilteredHeapObject`] for a managed object in a filtered
    /// heap, [`FormatError::HugeObjectNotFound`] if an indexed huge object is absent, and
    /// [`FormatError::UnexpectedEof`] if a managed object has no allocated block or a required
    /// file region is truncated. Returns [`FormatError::ChunkedReadError`] if indirect-block
    /// traversal exceeds 64 levels, [`FormatError::ValueTooLargeForPlatform`] if a value does not
    /// fit the platform's address width, and [`FormatError::OffsetOverflow`] if a file offset
    /// overflows. Returns errors from parsing heap blocks and huge-object index metadata.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if cached checksum validation or a cached huge-object index is reused
    /// through both this method and [`read_from_source`](Self::read_from_source).
    pub fn read(
        &mut self,
        file_data: &[u8],
        id: FractalHeapIdView<'_>,
    ) -> Result<Vec<u8>, FormatError> {
        match id.kind() {
            FractalHeapIdKind::Managed {
                heap_offset,
                object_length,
            } => self.read_managed_object(file_data, heap_offset, object_length),
            FractalHeapIdKind::Huge(reference) => self.read_huge(file_data, reference),
            FractalHeapIdKind::Tiny { bytes } => Ok(bytes.to_vec()),
        }
    }

    /// Reads the object identified by a parsed ID through a [`Source`].
    ///
    /// # Errors
    ///
    /// Returns the resolution errors of [`read`](Self::read), and the error `source` returns if
    /// a read fails.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if cached checksum validation or a cached huge-object index is reused
    /// through both this method and [`read`](Self::read).
    pub fn read_from_source<S: Source + ?Sized>(
        &mut self,
        source: &S,
        id: FractalHeapIdView<'_>,
    ) -> Result<Vec<u8>, FormatError> {
        match id.kind() {
            FractalHeapIdKind::Managed {
                heap_offset,
                object_length,
            } => self.read_managed_object_from_source(source, heap_offset, object_length),
            FractalHeapIdKind::Huge(reference) => self.read_huge_from_source(source, reference),
            FractalHeapIdKind::Tiny { bytes } => Ok(bytes.to_vec()),
        }
    }

    /// The `(address, length)` of huge object `huge_id`, parsing the heap's
    /// huge-object index out of `records` the first time one is asked for.
    ///
    /// What the index caches are file addresses, resolved against the image
    /// `backend` names. Every caller reads a whole heap through one backend, so
    /// a reader that saw both would be a caller that had changed which file it
    /// meant mid-walk — addresses applied to the wrong image read the wrong
    /// bytes and report no error, which is worth an assertion where the two meet.
    fn locate_huge<F>(
        &mut self,
        huge_id: u64,
        backend: Backend,
        records: F,
    ) -> Result<(StoredAddress, u64), FormatError>
    where
        F: FnOnce() -> Result<Vec<BTreeV2Record>, FormatError>,
    {
        if self.huge.is_none() {
            self.huge = Some(HugeObjectIndex::decode(
                &records()?,
                self.offset_size,
                self.length_size,
            )?);
            self.huge_backend = Some(backend);
        }
        debug_assert_eq!(
            self.huge_backend,
            Some(backend),
            "a heap's huge-object index holds addresses into the image it was decoded from",
        );
        self.huge
            .as_ref()
            .expect("filled immediately above")
            .locate(huge_id)
    }

    /// Reads a huge object at its inline location or through the cached index.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`read`](Self::read) for a huge object.
    fn read_huge(
        &mut self,
        file_data: &[u8],
        reference: HugeObjectReference,
    ) -> Result<Vec<u8>, FormatError> {
        let (addr, len) = match reference {
            HugeObjectReference::Inline { addr, len } => (addr, len),
            HugeObjectReference::Indexed(huge_id) => {
                let (offset_size, length_size) = (self.offset_size, self.length_size);
                let btree_addr = self.header.btree_huge_objects_address.get().to_usize()?;
                self.locate_huge(huge_id, Backend::Buffered, || {
                    let header =
                        BTreeV2Header::parse(file_data, btree_addr, offset_size, length_size)?;
                    check_huge_object_btree(&header, offset_size, length_size)?;
                    collect_btree_v2_records(file_data, &header, offset_size, length_size)
                })?
            }
        };
        slice_object(file_data, addr.get(), len.to_usize()?)
    }

    /// Reads a huge object through a [`Source`] at its inline location or through the cached index.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`read_from_source`](Self::read_from_source) for a huge object.
    fn read_huge_from_source<S: Source + ?Sized>(
        &mut self,
        source: &S,
        reference: HugeObjectReference,
    ) -> Result<Vec<u8>, FormatError> {
        let (addr, len) = match reference {
            HugeObjectReference::Inline { addr, len } => (addr, len),
            HugeObjectReference::Indexed(huge_id) => {
                let (offset_size, length_size) = (self.offset_size, self.length_size);
                let btree_addr = self.header.btree_huge_objects_address.get();
                self.locate_huge(huge_id, Backend::Streaming, || {
                    let header = BTreeV2Header::parse_from_source(
                        &SourceMetadata(source),
                        btree_addr,
                        offset_size,
                        length_size,
                    )?;
                    check_huge_object_btree(&header, offset_size, length_size)?;
                    collect_btree_v2_records_from_source(source, &header, offset_size, length_size)
                })?
            }
        };
        read_object_at_source(source, addr.get(), len.to_usize()?)
    }

    fn read_managed_object(
        &mut self,
        file_data: &[u8],
        heap_offset: u64,
        object_length: u64,
    ) -> Result<Vec<u8>, FormatError> {
        let image = FileRegion::new(
            FileRange::new(
                self.addresses.resolve(StoredAddress::ZERO)?,
                file_data.len() as u64,
            )?,
            file_data,
        )?;
        self.read_managed_object_with(
            |range| buffered_region(image, range).map(|region| Cow::Borrowed(region.bytes())),
            file_data.len() as u64,
            Backend::Buffered,
            heap_offset,
            object_length,
        )
    }

    fn read_managed_object_from_source(
        &mut self,
        source: &(impl Source + ?Sized),
        heap_offset: u64,
        object_length: u64,
    ) -> Result<Vec<u8>, FormatError> {
        let addresses = self.addresses;
        self.read_managed_object_with(
            |range| read_header_range(source, addresses, range).map(Cow::Owned),
            source.len(),
            Backend::Streaming,
            heap_offset,
            object_length,
        )
    }

    fn read_managed_object_with<'bytes>(
        &mut self,
        read: impl Fn(FileRange) -> Result<Cow<'bytes, [u8]>, FormatError>,
        source_len: u64,
        backend: Backend,
        heap_offset: u64,
        object_length: u64,
    ) -> Result<Vec<u8>, FormatError> {
        if self.header.filtering.is_filtered() {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        if self.validated_direct.is_empty() {
            self.validated_backend = Some(backend);
        }
        debug_assert_eq!(
            self.validated_backend,
            Some(backend),
            "validated direct blocks belong to the backend that supplied their bytes"
        );
        let blocks = self.managed_layout()?;
        let mut block = blocks.root()?.ok_or(FormatError::UnexpectedEof {
            expected: 1,
            available: 0,
        })?;
        for _depth in 0..64 {
            match block {
                FractalHeapBlock::Direct(direct) => {
                    let reference = direct.at(self.addresses.resolve(direct.address())?)?;
                    let require_checksum = self.access_mode == AccessMode::ReadWrite
                        && self.header.flags
                            & hdf5_pure_format::__private::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED
                            != 0
                        && !self.validated_direct.contains(&direct.address());
                    let range = if require_checksum {
                        reference.parse_range()
                    } else {
                        reference.object_prefix_range()?
                    };
                    let bytes = read(range)?;
                    let region = FileRegion::new(range, &bytes)?;
                    let object = reference.parse_object(region, heap_offset, object_length)?;
                    if require_checksum {
                        reference.parse_checked(region)?;
                        self.validated_direct.insert(direct.address());
                        return Ok(object.parse(region)?.to_vec());
                    }
                    return Ok(read(object.range())?.into_owned());
                }
                FractalHeapBlock::Indirect(indirect) => {
                    let reference = indirect.at(self.addresses.resolve(indirect.address())?)?;
                    let range = reference.parse_range();
                    let bytes = read(range)?;
                    let view = reference.parse(FileRegion::new(range, &bytes)?)?;
                    let mut selected = None;
                    for child in view.children() {
                        let child = child?;
                        let (start, length) = match child {
                            FractalHeapBlock::Direct(direct) => {
                                (direct.heap_offset(), direct.decoded_size())
                            }
                            FractalHeapBlock::Indirect(indirect) => (
                                indirect.heap_offset(),
                                blocks
                                    .layout()
                                    .indirect_heap_size(indirect.rows())
                                    .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?,
                            ),
                        };
                        if heap_offset >= start && heap_offset - start < length {
                            selected = Some(child);
                            break;
                        }
                    }
                    block = selected.ok_or(FormatError::UnexpectedEof {
                        expected: heap_offset
                            .to_usize()?
                            .saturating_add(object_length.to_usize()?),
                        available: source_len.to_usize().unwrap_or(usize::MAX),
                    })?;
                }
            }
        }
        Err(FormatError::ChunkedReadError(
            "fractal heap: maximum recursion depth exceeded".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use core::cell::RefCell;
    use hdf5_pure_format::__private::FilterDescription;
    use hdf5_pure_format::__private::FilterPipeline;
    use hdf5_pure_format::__private::FractalHeapFiltering;
    use hdf5_pure_format::__private::FractalHeapHeaderFrame;
    use hdf5_pure_format::__private::FractalHeapIdLayout;
    use rstest::rstest;
    use test_util::fractal_heap;
    use test_util::widths::Widths;

    use crate::address::{BaseAddress, BaseAddressExt, OpenedImageContext};
    use crate::source::{BaseOffsetSource, BytesSource};

    use super::*;

    // "The Data Storage - Filter Pipeline Message", format specification version 4.0, defines
    // pipeline version 2.
    const PIPELINE_VERSION_TWO: u8 = 2;

    #[rstest]
    #[case::plain(0, 0)]
    #[case::userblock(512, 512)]
    #[case::relocated(512, 1024)]
    #[case::relocated_back(512, 0)]
    fn source_header_reads_use_the_effective_physical_base(
        #[case] encoded_base: u64,
        #[case] effective_base: u64,
        #[values(0, 3)] root_rows: u16,
        #[values(0, 2)] huge_objects: u64,
    ) {
        let mut header = FractalHeapHeader::parse_bytes(
            &fractal_heap::Header::new(256).build(Widths::EIGHT),
            OffsetWidth::Eight,
            LengthWidth::Eight,
        )
        .unwrap();
        header.current_rows_in_root_indirect_block = root_rows;
        header.huge_objects_count = huge_objects;
        header.huge_objects_size = huge_objects * 1024;
        let bytes = header
            .serialize(OffsetWidth::Eight, LengthWidth::Eight)
            .unwrap();
        let header_address = 13;
        let mut image = vec![0xFF; usize::try_from(effective_base + header_address).unwrap()];
        image.extend_from_slice(&bytes);
        let source = HeaderSource {
            bytes: &image,
            reads: RefCell::new(Vec::new()),
        };
        let addresses = OpenedImageContext::reconcile(
            BaseAddress::new(encoded_base),
            encoded_base + header_address + bytes.len() as u64,
            effective_base,
        )
        .unwrap()
        .addresses();
        let framed = BaseOffsetSource {
            inner: &source,
            base: addresses.effective_base(),
        };
        assert_eq!(
            read_header_from_source(&framed, addresses, StoredAddress::new(header_address), 8, 8)
                .unwrap(),
            header
        );
        assert_eq!(
            *source.reads.borrow(),
            vec![
                (
                    effective_base + header_address,
                    FractalHeapHeaderFrame::PREFIX_LEN
                ),
                (effective_base + header_address, bytes.len())
            ]
        );
    }

    #[rstest]
    #[case::prefix(8, 9)]
    #[case::header(145, 146)]
    fn a_source_short_read_remains_a_source_error(
        #[case] available: usize,
        #[case] expected: usize,
    ) {
        let mut bytes = fractal_heap::Header::new(256).build(Widths::EIGHT);
        bytes.truncate(available);
        assert_eq!(
            read_header_from_source(
                &BytesSource::new(&bytes),
                AddressContext::from_effective_base(BaseAddress::ZERO),
                StoredAddress::ZERO,
                8,
                8
            )
            .unwrap_err(),
            FormatError::UnexpectedEof {
                expected,
                available
            }
        );
    }

    struct HeaderSource<'a> {
        bytes: &'a [u8],
        reads: RefCell<Vec<(u64, usize)>>,
    }

    impl Source for HeaderSource<'_> {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }

        fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
            BytesSource::new(self.bytes).read_at(offset, bytes)
        }

        fn read_metadata_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, FormatError> {
            self.reads.borrow_mut().push((offset, length));
            BytesSource::new(self.bytes).read_metadata_at(offset, length)
        }
    }

    #[rstest]
    #[case::read_only(AccessMode::ReadOnly)]
    #[case::read_write(AccessMode::ReadWrite)]
    fn checksummed_object_reads_follow_the_access_mode(#[case] mode: AccessMode) {
        let mut image = fractal_heap::heap_with_one_object(b"abc", Widths::EIGHT);
        let mut header = FractalHeapHeader::parse(&image, 0, 8, 8).unwrap();
        header.flags = hdf5_pure_format::__private::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
        let at = header.root_block_address.get() as usize;
        image.resize(at + 128, 0);
        image[at + 15..at + 19].fill(0);
        image[at + 19..at + 22].copy_from_slice(b"abc");
        let checksum = crate::checksum::jenkins_lookup3(&image[at..at + 128]);
        image[at + 15..at + 19].copy_from_slice(&checksum.to_le_bytes());
        let source = HeaderSource {
            bytes: &image,
            reads: RefCell::new(Vec::new()),
        };
        let id = [0, 19, 0, 3, 0, 0, 0];
        let mut reader = HeapObjectReader::new(
            &header,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::ZERO,
            mode,
            8,
            8,
        );
        assert_eq!(
            reader
                .read_from_source(&source, parse_id(&header, &id))
                .unwrap(),
            b"abc"
        );
        assert_eq!(
            reader
                .read_from_source(&source, parse_id(&header, &id))
                .unwrap(),
            b"abc"
        );
        assert_eq!(
            *source.reads.borrow(),
            if mode == AccessMode::ReadWrite {
                vec![(at as u64, 128), (at as u64, 19), (at as u64 + 19, 3)]
            } else {
                vec![
                    (at as u64, 19),
                    (at as u64 + 19, 3),
                    (at as u64, 19),
                    (at as u64 + 19, 3),
                ]
            }
        );
    }

    #[test]
    fn short_source_reads_and_short_supplied_object_prefixes_are_distinct() {
        let image = fractal_heap::heap_with_one_object(b"abc", Widths::EIGHT);
        let header = FractalHeapHeader::parse(&image, 0, 8, 8).unwrap();
        let blocks = FractalHeapBlockLayout::new(
            &header,
            OffsetWidth::Eight,
            LengthWidth::Eight,
            StoredAddress::ZERO,
        )
        .unwrap();
        let FractalHeapBlock::Direct(block) = blocks.root().unwrap().unwrap() else {
            panic!("expected direct root")
        };
        let reference = block
            .at(AddressContext::from_effective_base(BaseAddress::ZERO)
                .resolve(block.address())
                .unwrap())
            .unwrap();
        let range = reference.object_prefix_range().unwrap();
        let supplied = FileRange::new(range.start(), range.length() - 1).unwrap();
        let bytes = &image[range.start().get() as usize..supplied.end().get() as usize];
        assert_eq!(
            reference
                .parse_object(FileRegion::new(supplied, bytes).unwrap(), 15, 3)
                .unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: range.start().get(),
                required_length: range.length(),
                provided_start: supplied.start().get(),
                provided_length: supplied.length()
            }
        );
        let source = BytesSource::new(&image[..supplied.end().get() as usize]);
        let id = [0, 15, 0, 3, 0, 0, 0];
        assert_eq!(
            HeapObjectReader::new(
                &header,
                AddressContext::from_effective_base(BaseAddress::ZERO),
                StoredAddress::ZERO,
                AccessMode::ReadOnly,
                8,
                8
            )
            .read_from_source(&source, parse_id(&header, &id))
            .unwrap_err(),
            FormatError::UnexpectedEof {
                expected: range.end().get() as usize,
                available: supplied.end().get() as usize
            }
        );
    }

    #[rstest]
    #[case::tiny(vec![TINY_ID_FIRST_BYTE | 2, b'a', b'b', b'c', 0, 0, 0, 0], false)]
    #[case::filtered_tiny(vec![TINY_ID_FIRST_BYTE | 2, b'a', b'b', b'c', 0, 0, 0, 0], true)]
    #[case::inline_huge(vec![HUGE_ID_FIRST_BYTE, 0x20, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0], false)]
    fn a_parsed_inline_id_resolves_its_object(
        #[case] bytes: Vec<u8>,
        #[case] filtered: bool,
        #[values(false, true)] streaming: bool,
    ) {
        let mut header = FractalHeapHeader::parse(
            &fractal_heap::Header::new(256).build(Widths::EIGHT),
            0,
            8,
            8,
        )
        .unwrap();
        header.heap_id_length = u16::try_from(bytes.len()).unwrap();
        if filtered {
            header.filtering = FractalHeapFiltering::Filtered {
                root_direct_block_size: 91,
                root_filter_mask: 0,
                pipeline: FilterPipeline {
                    version: PIPELINE_VERSION_TWO,
                    filters: vec![FilterDescription {
                        filter_id: crate::filter_pipeline::FILTER_DEFLATE,
                        name: None,
                        flags: 0,
                        client_data: vec![6],
                    }],
                },
            };
        }
        let id = parse_id(&header, &bytes);
        let mut image = vec![0; 64];
        image[0x20..0x23].copy_from_slice(b"abc");
        let mut reader = HeapObjectReader::new(
            &header,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::ZERO,
            AccessMode::ReadOnly,
            8,
            8,
        );
        let object = if streaming {
            reader.read_from_source(&BytesSource::new(image), id)
        } else {
            reader.read(&image, id)
        }
        .unwrap();
        assert_eq!(object, b"abc");
    }

    #[test]
    fn read_managed_object_from_direct_block() {
        let file_data = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);
        let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();

        // Build heap ID for the test data written in build_simple_heap.
        // The test data "Hello, World!" is at the data area of the direct block.
        // The direct block header is 5 + 8 + 2 = 15 bytes (for max_heap_size=16, ceil(16/8)=2).
        // Wait, max_heap_size=16, ceil(16/8)=2. Header = sig(4)+ver(1)+addr(8)+bo(2) = 15.
        // The data was placed at data_start = block_addr + 15.
        // Since offset is from block start, the object is at offset 15 within the block.
        let dblock_header_size = 5 + 8 + (hdr.max_heap_size as usize).div_ceil(8); // 15
        let offset: u64 = dblock_header_size as u64;
        let length: u64 = 13;
        let payload = offset | (length << hdr.max_heap_size);
        let mut id = vec![0u8; 7];
        id[0] = 0x00;
        for i in 0..6 {
            id[1 + i] = ((payload >> (i * 8)) & 0xFF) as u8;
        }

        let obj = HeapObjectReader::new(
            &hdr,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::ZERO,
            AccessMode::ReadOnly,
            8,
            8,
        )
        .read(&file_data, parse_id(&hdr, &id))
        .unwrap();
        assert_eq!(&obj, b"Hello, World!");
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_managed_object_matches_buffered() {
        use crate::source::{BytesSource, ReadSeekSource};
        let file_data = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);

        // Same heap ID as `read_managed_object_from_direct_block`.
        let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();
        let dblock_header_size = 5 + 8 + (hdr.max_heap_size as usize).div_ceil(8);
        let payload = (dblock_header_size as u64) | (13u64 << hdr.max_heap_size);
        let mut id = vec![0u8; 7];
        for i in 0..6 {
            id[1 + i] = ((payload >> (i * 8)) & 0xFF) as u8;
        }

        let buffered = HeapObjectReader::new(
            &hdr,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::ZERO,
            AccessMode::ReadOnly,
            8,
            8,
        )
        .read(&file_data, parse_id(&hdr, &id))
        .unwrap();

        // Header parsed from a source, then the object fetched from a source.
        let mem = BytesSource::new(&file_data);
        let hdr_mem = read_header_from_source(
            &mem,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::ZERO,
            8,
            8,
        )
        .unwrap();
        let from_mem = HeapObjectReader::new(
            &hdr_mem,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::ZERO,
            AccessMode::ReadOnly,
            8,
            8,
        )
        .read_from_source(&mem, parse_id(&hdr_mem, &id))
        .unwrap();

        let seek = ReadSeekSource::new(std::io::Cursor::new(file_data)).unwrap();
        let hdr_seek = read_header_from_source(
            &seek,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::ZERO,
            8,
            8,
        )
        .unwrap();
        let from_seek = HeapObjectReader::new(
            &hdr_seek,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::ZERO,
            AccessMode::ReadOnly,
            8,
            8,
        )
        .read_from_source(&seek, parse_id(&hdr_seek, &id))
        .unwrap();

        assert_eq!(buffered, from_mem);
        assert_eq!(buffered, from_seek);
        assert_eq!(&from_seek, b"Hello, World!");
    }

    #[test]
    fn filtered_managed_heap_is_refused_not_misparsed() {
        // A filtered managed heap stores its direct-block contents filter-encoded
        // and inserts per-child `filtered_size`/`filter_mask` fields in indirect
        // blocks. We do not decode either, so every managed read path must refuse
        // cleanly (UnsupportedFilteredHeapObject) rather than return raw bytes or
        // mis-stride the child-pointer walk.
        let managed_id = [0x00u8, 0, 0, 0, 0, 0, 0]; // type bits 0 -> Managed

        // Root is a direct block.
        let mut h = FractalHeapHeader {
            heap_id_length: 7,
            filtering: FractalHeapFiltering::Filtered {
                root_direct_block_size: 91,
                root_filter_mask: 1,
                pipeline: FilterPipeline {
                    version: PIPELINE_VERSION_TWO,
                    filters: vec![],
                },
            },
            flags: 0,
            max_managed_object_size: 0,
            next_huge_object_id: 0,
            btree_huge_objects_address: StoredAddress::new(u64::MAX),
            free_space_in_managed_blocks: 0,
            managed_block_free_space_manager_address: StoredAddress::new(u64::MAX),
            managed_space: 0,
            allocated_managed_space: 0,
            direct_block_allocation_iterator_offset: 0,
            managed_objects_count: 0,
            huge_objects_size: 0,
            huge_objects_count: 0,
            tiny_objects_size: 0,
            tiny_objects_count: 0,
            table_width: 4,
            starting_block_size: 512,
            max_direct_block_size: 65536,
            max_heap_size: 64,
            start_root_rows: 1,
            root_block_address: StoredAddress::new(0x100),
            current_rows_in_root_indirect_block: 0,
        };
        let file = vec![0u8; 0x400];
        assert_eq!(
            HeapObjectReader::new(
                &h,
                AddressContext::from_effective_base(BaseAddress::ZERO),
                StoredAddress::ZERO,
                AccessMode::ReadOnly,
                8,
                8
            )
            .read(&file, parse_id(&h, &managed_id)),
            Err(FormatError::UnsupportedFilteredHeapObject)
        );

        // Root is an indirect block: the refusal happens before any child walk.
        h.current_rows_in_root_indirect_block = 2;
        assert_eq!(
            HeapObjectReader::new(
                &h,
                AddressContext::from_effective_base(BaseAddress::ZERO),
                StoredAddress::ZERO,
                AccessMode::ReadOnly,
                8,
                8
            )
            .read(&file, parse_id(&h, &managed_id)),
            Err(FormatError::UnsupportedFilteredHeapObject)
        );
    }

    /// Type-1 record: address(8) + length(8) + id(8), little-endian.
    fn huge_record(addr: u64, len: u64, id: u64) -> BTreeV2Record {
        let mut d = Vec::new();
        d.extend_from_slice(&addr.to_le_bytes());
        d.extend_from_slice(&len.to_le_bytes());
        d.extend_from_slice(&id.to_le_bytes());
        BTreeV2Record { data: d }
    }

    #[test]
    fn huge_object_index_locates_by_id() {
        let records = vec![
            huge_record(0x1000, 5000, 1),
            huge_record(0x2000, 6000, 2),
            huge_record(0x3000, 7000, 5),
        ];
        let index = HugeObjectIndex::decode(&records, 8, 8).unwrap();
        assert_eq!(index.locate(2).unwrap(), (StoredAddress::new(0x2000), 6000));
        assert_eq!(index.locate(5).unwrap(), (StoredAddress::new(0x3000), 7000));
        assert_eq!(
            index.locate(9),
            Err(FormatError::HugeObjectNotFound(9)),
            "an id no record carries is not found rather than mismatched"
        );
    }

    /// The lookup is a binary search, so the index has to impose the ordering
    /// itself rather than inherit whatever order the records arrived in.
    #[test]
    fn huge_object_index_orders_records_it_receives_out_of_order() {
        let records = vec![
            huge_record(0x3000, 7000, 5),
            huge_record(0x1000, 5000, 1),
            huge_record(0x2000, 6000, 2),
        ];
        let index = HugeObjectIndex::decode(&records, 8, 8).unwrap();
        for (id, want) in [
            (1, (StoredAddress::new(0x1000), 5000)),
            (2, (StoredAddress::new(0x2000), 6000)),
            (5, (StoredAddress::new(0x3000), 7000)),
        ] {
            assert_eq!(index.locate(id).unwrap(), want);
        }
    }

    /// A huge-objects B-tree that is not the type-1 layout is refused before its
    /// records are read as that layout.
    ///
    /// Type 2 is the filtered counterpart, whose records are longer and put the
    /// object ID somewhere else: decoding one as type 1 reads an ID out of the
    /// filter mask and returns some other object, or none, without complaint.
    #[test]
    fn a_huge_object_btree_of_another_type_is_refused() {
        let btree = |tree_type: u8, record_size: u16| BTreeV2Header {
            tree_type,
            node_size: 512,
            record_size,
            depth: 0,
            root_node_address: StoredAddress::new(0x100),
            num_records_in_root: 1,
            total_records: 1,
        };

        assert_eq!(
            check_huge_object_btree(&btree(2, 36), 8, 8),
            Err(FormatError::UnexpectedHugeObjectBTree {
                tree_type: 2,
                record_size: 36,
                required: 24,
            }),
            "a filtered huge-object tree is long enough to pass a length check alone"
        );
        assert_eq!(
            check_huge_object_btree(&btree(1, 16), 8, 8),
            Err(FormatError::UnexpectedHugeObjectBTree {
                tree_type: 1,
                record_size: 16,
                required: 24,
            }),
            "records too short for the layout the type claims"
        );
        assert_eq!(check_huge_object_btree(&btree(1, 24), 8, 8), Ok(()));
    }

    /// A record too short for the type-1 layout is a heap this reader has
    /// misread, not one object to skip past. The backstop behind
    /// [`check_huge_object_btree`], which refuses such a tree by its declared
    /// record size before any record reaches the decode.
    #[test]
    fn huge_object_index_refuses_a_truncated_record() {
        let records = vec![BTreeV2Record {
            data: vec![0u8; 20],
        }];
        assert!(matches!(
            HugeObjectIndex::decode(&records, 8, 8),
            Err(FormatError::UnexpectedEof { .. })
        ));
    }

    /// One reader parses the heap's huge-object index once, however many huge
    /// objects are read through it, on both backends.
    ///
    /// Parsing it per object returns exactly the same bytes while making a walk
    /// over the heap quadratic in the number of huge objects, so only the parse
    /// count separates the two.
    #[test]
    fn a_reader_parses_the_huge_object_index_once() {
        const COUNT: u64 = 5;

        // Attributes too large for a managed heap object, so the heap holds
        // nothing but huge ones.
        let mut builder = crate::FileBuilder::new();
        for i in 0..COUNT {
            builder.set_attr(
                &format!("a{i}"),
                crate::AttrValue::StringArray(vec![format!("{i:0700}"); 100]),
            );
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);
        let bytes = builder.finish().unwrap();

        let frhp = bytes
            .windows(4)
            .position(|w| w == b"FRHP")
            .expect("dense storage writes a fractal heap");
        let heap = FractalHeapHeader::parse(&bytes, frhp, 8, 8).unwrap();

        // A huge heap ID: type 1 in bits 4-5 of byte 0, then the object's id,
        // little-endian, across the rest of the ID. The writer's own encoding,
        // which lets this reach the objects without walking the name index.
        let heap_id = |id: u64| {
            let mut bytes = vec![0u8; heap.heap_id_length as usize];
            bytes[0] = 0x10;
            for (i, slot) in bytes[1..].iter_mut().take(8).enumerate() {
                *slot = ((id >> (i * 8)) & 0xFF) as u8;
            }
            bytes
        };
        let ids: Vec<Vec<u8>> = (1..=COUNT).map(heap_id).collect();

        reset_huge_index_decodes();
        let mut reader = HeapObjectReader::new(
            &heap,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::new(frhp as u64),
            AccessMode::ReadOnly,
            8,
            8,
        );
        let buffered: Vec<Vec<u8>> = ids
            .iter()
            .map(|id| reader.read(&bytes, parse_id(&heap, id)).unwrap())
            .collect();
        assert_eq!(
            huge_index_decodes(),
            1,
            "buffered reads re-parsed the index"
        );

        reset_huge_index_decodes();
        let source = BytesSource::new(bytes.clone());
        let mut reader = HeapObjectReader::new(
            &heap,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::new(frhp as u64),
            AccessMode::ReadOnly,
            8,
            8,
        );
        let streamed: Vec<Vec<u8>> = ids
            .iter()
            .map(|id| {
                reader
                    .read_from_source(&source, parse_id(&heap, id))
                    .unwrap()
            })
            .collect();
        assert_eq!(
            huge_index_decodes(),
            1,
            "streaming reads re-parsed the index"
        );

        assert_eq!(buffered, streamed);
        assert_eq!(buffered.len(), COUNT as usize);
        for (i, object) in buffered.iter().enumerate() {
            assert!(
                object.windows(2).any(|w| w == format!("a{i}").as_bytes()),
                "object {i} is not the attribute message it should be"
            );
        }
    }

    fn parse_id<'a>(header: &FractalHeapHeader, bytes: &'a [u8]) -> FractalHeapIdView<'a> {
        FractalHeapIdLayout::new(header, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(bytes)
            .unwrap()
    }

    // A version 0 huge ID has type 1 in bits 4 and 5 ("Fractal Heap", specification 4.0).
    const HUGE_ID_FIRST_BYTE: u8 = 0x10;
    // A version 0 tiny ID has type 2 in bits 4 and 5 ("Fractal Heap", specification 4.0).
    const TINY_ID_FIRST_BYTE: u8 = 0x20;
}
