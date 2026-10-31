//! Reads of global heap objects, such as the bytes of variable-length values.
//!
//! [`GlobalHeapReader`] walks each collection once, at the first read of one of its objects, and
//! reads the object headers as metadata, several headers to a read where they sit close together.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::vec::Vec;
use core::num::NonZeroU16;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::MetadataBytes;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
use hdf5_pure_format::__private::GlobalHeapObject;
use hdf5_pure_format::__private::GlobalHeapPayloadRef;

use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::Source;
use crate::width::LengthWidth;

/// A global heap ID, with the address of its collection resolved to a physical position.
///
/// The ID is defined in "Global Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct GlobalHeapId {
    pub(crate) collection: PhysicalAddress,
    pub(crate) index: NonZeroU16,
}

/// A reader of the data of the global heap objects a caller passes to [`new`](Self::new).
///
/// The reader walks a collection at the first read of one of its objects and keeps the range of
/// the data of each object the caller passed. The walk parses every object header of the
/// collection, so a read rejects a collection that is malformed past the object it reads.
pub(crate) struct GlobalHeapReader<'s, S: Source + ?Sized> {
    source: &'s S,
    lengths: LengthWidth,
    directories: BTreeMap<PhysicalAddress, Directory>,
}

impl<'s, S: Source + ?Sized> GlobalHeapReader<'s, S> {
    /// Creates a reader of the objects at `ids` in `source`, in collections whose length fields
    /// are `lengths` wide.
    pub(crate) fn new(
        source: &'s S,
        lengths: LengthWidth,
        ids: impl IntoIterator<Item = GlobalHeapId>,
    ) -> Self {
        let mut directories: BTreeMap<PhysicalAddress, Directory> = BTreeMap::new();
        for GlobalHeapId { collection, index } in ids {
            directories
                .entry(collection)
                .or_insert_with(|| Directory {
                    objects: Vec::new(),
                    walked: false,
                })
                .objects
                .push((index, None));
        }
        for directory in directories.values_mut() {
            directory.objects.sort_unstable_by_key(|&(index, _)| index);
            directory.objects.dedup_by_key(|&mut (index, _)| index);
        }
        Self {
            source,
            lengths,
            directories,
        }
    }

    /// Reads the first `len` bytes of the data of the object at `id`.
    ///
    /// The reader reads the object headers through [`Source::read_metadata_at`] and the data
    /// through [`Source::read_exact_at`], so a metadata cache does not hold the data.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the caller did not pass `id` to [`new`](Self::new),
    /// [`FormatError::GlobalHeapObjectNotFound`] if the collection does not hold an object at the
    /// index, [`FormatError::VlDataError`] if `len` exceeds the size of the object, the errors
    /// [`Directory::walk`] returns, [`FormatError::ValueTooLargeForPlatform`] if `len` is above
    /// [`usize::MAX`], and the error the source returns if the read of the data fails.
    pub(crate) fn read(&mut self, id: GlobalHeapId, len: u64) -> Result<Vec<u8>, FormatError> {
        let payload = self.payload(id)?.range();
        if len > payload.length() {
            return Err(FormatError::VlDataError(format!(
                "a {len}-byte read exceeds global heap object size {}",
                payload.length()
            )));
        }
        self.source
            .read_exact_at(payload.start().get(), len.to_usize()?)
    }

    fn payload(
        &mut self,
        GlobalHeapId { collection, index }: GlobalHeapId,
    ) -> Result<GlobalHeapPayloadRef, FormatError> {
        let unregistered = || {
            FormatError::Internal(format!(
                "global heap object {index} of the collection at {} is not registered",
                collection.get()
            ))
        };
        let directory = self
            .directories
            .get_mut(&collection)
            .ok_or_else(unregistered)?;
        if !directory.walked {
            directory.walk(self.source, self.lengths, collection)?;
        }
        match directory
            .objects
            .binary_search_by_key(&index, |&(index, _)| index)
        {
            Ok(at) => directory.objects[at]
                .1
                .ok_or(FormatError::GlobalHeapObjectNotFound {
                    collection_address: collection.get(),
                    index: index.get(),
                }),
            Err(_) => Err(unregistered()),
        }
    }
}

/// The objects a caller passed to [`GlobalHeapReader::new`] in one collection.
struct Directory {
    /// The indices in ascending order, each with the range of its object's data, or `None` before
    /// the walk and where the collection does not hold an object at the index.
    objects: Vec<(NonZeroU16, Option<GlobalHeapPayloadRef>)>,
    /// Whether a walk of the collection has parsed every object header.
    walked: bool,
}

impl Directory {
    /// Walks the collection at `address` and records the range of the data of each object at one
    /// of the indices.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedSourceEof`] if the collection runs past the end of `source`, the
    /// errors [`GlobalHeapCollectionLocator::parse`] and
    /// [`GlobalHeapWalk::parse`](hdf5_pure_format::__private::GlobalHeapWalk::parse) return, and
    /// the errors [`read`] returns.
    fn walk(
        &mut self,
        source: &(impl Source + ?Sized),
        lengths: LengthWidth,
        address: PhysicalAddress,
    ) -> Result<(), FormatError> {
        walk_collection(source, lengths, address, |object| {
            if let Ok(at) = self
                .objects
                .binary_search_by_key(&object.index(), |&(index, _)| index)
            {
                self.objects[at].1 = Some(object.payload());
            }
        })?;
        self.walked = true;
        Ok(())
    }
}

/// Walks the global heap collection at `address`, passes each object other than the free space
/// to `visit`, and returns the range of the collection.
///
/// The walk parses every object header of the collection, in the order the collection stores
/// them.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedSourceEof`] if the collection runs past the end of `source`, the
/// errors [`GlobalHeapCollectionLocator::parse`] and
/// [`GlobalHeapWalk::parse`](hdf5_pure_format::__private::GlobalHeapWalk::parse) return, and the
/// errors [`read`] returns.
pub(crate) fn walk_collection(
    source: &(impl Source + ?Sized),
    lengths: LengthWidth,
    address: PhysicalAddress,
    mut visit: impl FnMut(GlobalHeapObject),
) -> Result<FileRange, FormatError> {
    let locator = GlobalHeapCollectionLocator::new(address, lengths);
    let header_range = locator.parse_range()?;
    let header = read(source, header_range)?;
    let collection = locator.parse(FileRegion::new(header_range, &header)?)?;
    let end = collection.range().end().get();
    if end > source.len() {
        return Err(FormatError::UnexpectedSourceEof {
            expected: end,
            available: source.len(),
        });
    }

    let mut window = HeaderWindow::new(collection.range());
    let mut walk = collection.walk();
    while let Some(range) = walk.parse_range() {
        if let Some(object) = walk.parse(window.region(source, range)?)? {
            visit(object);
        }
    }
    Ok(collection.range())
}

/// The bytes of one collection that a walk reads its object headers from.
struct HeaderWindow<'s> {
    /// The range of the collection, which no read runs past.
    collection: FileRange,
    /// The range and the bytes of the last read.
    held: Option<(FileRange, MetadataBytes<'s>)>,
    /// The start of the previous object header the walk requested, `None` before the first.
    previous_header: Option<PhysicalAddress>,
    /// The last [`STRIDE_HISTORY`] strides between the object headers the walk requested, the most
    /// recent last. A zero is a stride not yet recorded.
    strides: [u64; STRIDE_HISTORY],
}

impl<'s> HeaderWindow<'s> {
    fn new(collection: FileRange) -> Self {
        Self {
            collection,
            held: None,
            previous_header: None,
            strides: [0; STRIDE_HISTORY],
        }
    }

    /// Returns the region of `range`, and reads it from `source` where the bytes held do not
    /// contain it.
    ///
    /// # Errors
    ///
    /// Returns the errors [`read`] returns.
    fn region(
        &mut self,
        source: &'s (impl Source + ?Sized),
        range: FileRange,
    ) -> Result<FileRegion<'_>, FormatError> {
        if let Some(previous_header) = self.previous_header {
            self.strides.rotate_left(1);
            if let Some(newest) = self.strides.last_mut() {
                *newest = range.start().get().saturating_sub(previous_header.get());
            }
        }
        self.previous_header = Some(range.start());
        let held = match self.held.take() {
            Some((held, bytes)) if held.contains(range) => (held, bytes),
            _ => {
                let held = FileRange::new(range.start(), self.span(range))?;
                (held, read(source, held)?)
            }
        };
        let (held, bytes) = self.held.insert(held);
        FileRegion::new(*held, bytes)?.subregion(range)
    }

    /// Returns the number of bytes a read at the start of `range` reads.
    ///
    /// A read covers `range` and [`LOOKAHEAD`] strides past it, at the widest of the last
    /// [`STRIDE_HISTORY`] strides, up to [`WINDOW`] bytes and up to the end of the collection.
    /// Before the first stride, and where the widest stride and `range` together exceed a window,
    /// the read covers `range` alone.
    ///
    /// The span comes from the widest stride: in a collection whose object sizes alternate, the
    /// stride into a header is smallest where the stride out of it is largest, and with a span from
    /// the last stride the walk would read a window at the header of each large object and discard
    /// most of it.
    fn span(&self, range: FileRange) -> u64 {
        let need = range.length();
        let widest = self.strides.iter().copied().max().unwrap_or(0);
        let ahead = if widest.saturating_add(need) > WINDOW {
            need
        } else {
            widest
                .saturating_mul(LOOKAHEAD)
                .saturating_add(need)
                .min(WINDOW)
        };
        self.collection
            .end()
            .get()
            .saturating_sub(range.start().get())
            .min(ahead)
            .max(need)
    }
}

/// Reads the bytes of `range` through [`Source::read_metadata_at`].
///
/// # Errors
///
/// Returns [`FormatError::ValueTooLargeForPlatform`] if the length of `range` is above
/// [`usize::MAX`], and the error `source` returns if the read fails.
fn read(
    source: &(impl Source + ?Sized),
    range: FileRange,
) -> Result<MetadataBytes<'_>, FormatError> {
    source.read_metadata(range.start().get(), range.length().to_usize()?)
}

/// The most bytes one read of a walk reads.
const WINDOW: u64 = 4096;

/// The number of recent strides [`HeaderWindow::span`] takes the widest of.
///
/// After an object larger than a window, the walk reads one object header at a time for this many
/// headers.
const STRIDE_HISTORY: usize = 8;

/// The number of strides a read covers past the object header it needs.
const LOOKAHEAD: u64 = 32;

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use test_util::global_heap;
    use test_util::widths::Widths;

    use super::*;
    use crate::source::BytesSource;

    #[test]
    fn the_walk_reads_object_headers_as_metadata() {
        // The collection header, the first object header, and one window from the second on.
        let source = walk_for_the_last(&[b"hello", b"world"]);
        assert_eq!(
            (source.metadata_reads.get(), source.raw_reads.get()),
            (3, 1)
        );
    }

    #[test]
    fn the_walk_reads_512_small_object_headers_in_at_most_32_reads() {
        // 512 objects of 8 bytes: 512 * (16 + 8) = 12,288 bytes of headers and data, three
        // windows' worth.
        let source = walk_for_the_last(&[[0u8; 8].as_slice(); 512]);
        let reads = source.metadata_reads.get();
        assert!(
            reads <= 512 / 16,
            "walking 512 objects took {reads} reads, which scales with the object count rather \
             than the {} bytes of collection it covers",
            source.bytes.len()
        );
    }

    #[test]
    fn the_walk_reads_at_most_a_window_and_32_bytes_a_header_between_large_objects() {
        // 64 objects of 5,000 bytes, each stride wider than a window.
        let source = walk_for_the_last(&[[0u8; 5000].as_slice(); 64]);
        let read = source.metadata_bytes.get();
        let ceiling = WINDOW as usize + (64 + 2) * 32;
        assert!(
            read <= ceiling,
            "walking 64 objects of 5,000 bytes read {read} bytes of a {}-byte collection, \
             above the {ceiling} a header apiece costs",
            source.bytes.len()
        );
    }

    #[test]
    fn the_walk_reads_at_most_a_window_and_32_bytes_a_header_between_alternating_objects() {
        // Alternating 8-byte and 5,000-byte objects, so half the strides fit in a window.
        let small = [0u8; 8];
        let large = [0u8; 5000];
        let payloads: Vec<&[u8]> = (1..=256)
            .map(|i| if i % 2 == 0 { &large[..] } else { &small[..] })
            .collect();
        let source = walk_for_the_last(&payloads);
        let read = source.metadata_bytes.get();
        let ceiling = WINDOW as usize + (256 + 2) * 32;
        assert!(
            read <= ceiling,
            "walking 256 alternating objects read {read} bytes of a {}-byte collection, \
             above the {ceiling} a header apiece costs",
            source.bytes.len()
        );
    }

    /// Reads the last of `payloads` from a collection that holds them at the indices 1, 2, and so
    /// on, and returns the source with its counts of the reads.
    fn walk_for_the_last(payloads: &[&[u8]]) -> CountingSource {
        let objects: Vec<_> = (1..)
            .zip(payloads)
            .map(|(index, data)| global_heap::Object::new(index, data))
            .collect();
        let source = CountingSource {
            bytes: global_heap::collection(&objects, Widths::EIGHT),
            metadata_reads: Cell::new(0),
            metadata_bytes: Cell::new(0),
            raw_reads: Cell::new(0),
        };
        let last = GlobalHeapId {
            collection: PhysicalAddress::new(0),
            index: NonZeroU16::new(u16::try_from(payloads.len()).unwrap()).unwrap(),
        };
        let last_payload = payloads[payloads.len() - 1];
        assert_eq!(
            GlobalHeapReader::new(&source, LengthWidth::Eight, [last])
                .read(last, last_payload.len() as u64)
                .unwrap(),
            last_payload
        );
        source
    }

    /// A source over `bytes` that counts its reads.
    struct CountingSource {
        bytes: Vec<u8>,
        metadata_reads: Cell<usize>,
        metadata_bytes: Cell<usize>,
        raw_reads: Cell<usize>,
    }

    impl Source for CountingSource {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            self.raw_reads.set(self.raw_reads.get() + 1);
            BytesSource::new(&self.bytes).read_at(offset, buf)
        }

        fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
            self.metadata_reads.set(self.metadata_reads.get() + 1);
            self.metadata_bytes.set(self.metadata_bytes.get() + len);
            BytesSource::new(&self.bytes).read_exact_at(offset, len)
        }
    }
}
