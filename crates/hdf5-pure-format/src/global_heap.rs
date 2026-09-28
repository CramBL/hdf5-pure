//! The global heap collection parser and encoder.
//!
//! A collection holds global heap objects, such as the bytes of variable-length values, each at a
//! 1-based index. A global heap ID in the file holds the address of a collection and the index of
//! an object in it.

use alloc::vec::Vec;

use crate::bytes::read_length;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::metadata_source::MetadataSource;

/// The directory of a global heap collection: where each of its objects sits and how large it is.
///
/// The directory holds none of the object data, so a caller parses a collection once and reads
/// only the objects it looks up. The collection is defined in "Global Heap" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlobalHeapIndex {
    /// The objects, in the order the collection stores them.
    pub objects: Vec<GlobalHeapObjectInfo>,
}

/// The position and the size of one object in a global heap collection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlobalHeapObjectInfo {
    /// The index of the object in the collection, from 1. Index 0 is the free space, which
    /// [`GlobalHeapIndex`] leaves out.
    pub index: u16,
    /// The position of the object's data in the source, past its object header.
    pub data_address: u64,
    /// The size in bytes of the object's data, before the padding to a multiple of 8.
    pub size: u64,
}

/// Rounds `x` up to a multiple of 8, or returns [`FormatError::OffsetOverflow`] if that overflows.
fn pad8_u64(x: u64) -> Result<u64, FormatError> {
    x.checked_add(7)
        .map(|value| value & !7)
        .ok_or(FormatError::OffsetOverflow {
            offset: x,
            length: 7,
        })
}

/// The most bytes of a collection one refill of the directory walk reads, one 4 KiB page.
///
/// [`DirectoryWindow::get`] reads less where the headers sit close together, and one header alone
/// where they sit too far apart for one window to hold two of them.
const DIRECTORY_WINDOW: usize = 4096;

/// The number of recent strides a refill takes the widest of.
///
/// A stride is the distance between two object headers the walk requests. After an object larger
/// than a window, the walk reads one header at a time for this many objects.
const DIRECTORY_STRIDE_HISTORY: usize = 8;

/// The number of strides a refill reads past the header it needs, at the widest recent stride.
///
/// The bytes a refill reads ahead are proportional to the spacing of the headers, up to
/// [`DIRECTORY_WINDOW`].
const DIRECTORY_LOOKAHEAD: u64 = 32;

/// Consecutive bytes of one collection, which the directory walk reads its object headers from.
///
/// The window serves several headers from one read of the source. A refill reads through
/// [`MetadataSource::read_metadata_at`], so a source with a metadata cache may serve it from the
/// cache.
struct DirectoryWindow {
    /// The position in the source of the first byte of `bytes`.
    start: u64,
    /// The bytes the last refill read.
    bytes: Vec<u8>,
    /// The position of the previous request, `None` before the first.
    last: Option<u64>,
    /// The last [`DIRECTORY_STRIDE_HISTORY`] strides between requests, the most recent last. A zero
    /// is a stride not yet recorded, since the position the walk requests increases with every
    /// request.
    strides: [u64; DIRECTORY_STRIDE_HISTORY],
}

impl DirectoryWindow {
    const fn new() -> Self {
        Self {
            start: 0,
            bytes: Vec::new(),
            last: None,
            strides: [0; DIRECTORY_STRIDE_HISTORY],
        }
    }

    /// Returns the `need` bytes at `pos`, and refills from `source` where the window does not hold
    /// them.
    ///
    /// A refill reads `need` bytes and [`DIRECTORY_LOOKAHEAD`] strides past them, at the widest of
    /// the last [`DIRECTORY_STRIDE_HISTORY`] strides, up to [`DIRECTORY_WINDOW`] bytes. Where the
    /// widest stride and `need` exceed a window, a window cannot hold a second header, and the
    /// refill reads `need` bytes alone. Before the first stride the widest is zero, and the refill
    /// reads `need` bytes.
    ///
    /// The widest stride sets the span and the last one does not: in a collection whose object
    /// sizes alternate, the stride into a header is smallest where the stride out of it is
    /// largest, and a span set by the last stride reads a window at every change of size and
    /// discards most of it.
    ///
    /// The caller checks that `limit`, the end of the collection, lies within `source` and that
    /// `need` bytes fit before it, and a refill reads no further than `limit`.
    ///
    /// # Errors
    ///
    /// Returns the error `source` returns if a refill fails.
    fn get<'a>(
        &'a mut self,
        source: &(impl MetadataSource + ?Sized),
        pos: u64,
        need: usize,
        limit: u64,
    ) -> Result<&'a [u8], FormatError> {
        let held = pos
            .checked_sub(self.start)
            .and_then(|d| usize::try_from(d).ok())
            .filter(|at| {
                at.checked_add(need)
                    .is_some_and(|end| end <= self.bytes.len())
            });

        if let Some(last) = self.last {
            self.strides.rotate_left(1);
            self.strides[DIRECTORY_STRIDE_HISTORY - 1] = pos.saturating_sub(last);
        }
        self.last = Some(pos);

        let at = match held {
            Some(at) => at,
            None => {
                let widest = self.strides.iter().copied().max().unwrap_or(0);
                let ahead = if widest.saturating_add(need as u64) > DIRECTORY_WINDOW as u64 {
                    need
                } else {
                    widest
                        .saturating_mul(DIRECTORY_LOOKAHEAD)
                        .saturating_add(need as u64)
                        .min(DIRECTORY_WINDOW as u64)
                        .to_usize()
                        .unwrap_or(need)
                        .max(need)
                };
                // The span is at most `ahead`, a `usize`, so the narrowing cannot fail on a 32-bit
                // target and the fallback is unreachable.
                let span = limit
                    .saturating_sub(pos)
                    .min(ahead as u64)
                    .to_usize()
                    .unwrap_or(ahead)
                    .max(need);
                self.bytes = source.read_metadata_at(pos, span)?;
                self.start = pos;
                0
            }
        };

        Ok(&self.bytes[at..at + need])
    }
}

impl GlobalHeapIndex {
    /// Parses the directory of the global heap collection at `offset` in `source`.
    ///
    /// `length_size` is the superblock's "Size of Lengths" byte.
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse_filtered`](Self::parse_filtered) returns.
    pub fn parse(
        source: &(impl MetadataSource + ?Sized),
        offset: u64,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        Self::parse_filtered(source, offset, length_size, |_| true)
    }

    /// Parses the directory of the global heap collection at `offset` in `source`, and keeps the
    /// objects whose index `keep` accepts.
    ///
    /// The walk reads every object header, since each object's size gives the position of the
    /// next, and the directory holds only the objects `keep` accepts. A caller that looks up a few
    /// objects of a large collection, such as the strings of a few rows of a variable-length
    /// string dataset, keeps a directory of those objects alone in memory.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidGlobalHeapSignature`] if the collection does not begin with
    /// `GCOL`, [`FormatError::InvalidGlobalHeapVersion`] if its version is not 1,
    /// [`FormatError::InvalidLengthSize`] if `length_size` is not 2, 4, or 8,
    /// [`FormatError::VlDataError`] if the collection size is smaller than the collection header,
    /// [`FormatError::UnexpectedEof`] if the collection runs past the end of `source` or an object
    /// runs past the end of the collection, [`FormatError::OffsetOverflow`] if a position
    /// overflows a `u64`, and the error `source` returns if a read fails.
    pub fn parse_filtered(
        source: &(impl MetadataSource + ?Sized),
        offset: u64,
        length_size: u8,
        keep: impl Fn(u16) -> bool,
    ) -> Result<Self, FormatError> {
        let header_size = 8 + length_size as usize;
        let header = source.read_metadata_at(offset, header_size)?;

        if header[..4] != GCOL_SIGNATURE {
            return Err(FormatError::InvalidGlobalHeapSignature);
        }
        let version = header[4];
        if version != GCOL_VERSION {
            return Err(FormatError::InvalidGlobalHeapVersion(version));
        }

        let collection_size = read_length(&header, 8, length_size)?;
        if collection_size < header_size as u64 {
            return Err(FormatError::VlDataError(
                "global heap collection is smaller than its header".into(),
            ));
        }
        let collection_end =
            offset
                .checked_add(collection_size)
                .ok_or(FormatError::OffsetOverflow {
                    offset,
                    length: collection_size,
                })?;
        if collection_end > source.len() {
            return Err(FormatError::UnexpectedEof {
                expected: collection_end.to_usize().unwrap_or(usize::MAX),
                available: source.len().to_usize().unwrap_or(usize::MAX),
            });
        }

        let object_header_size = 8 + length_size as usize;
        let mut pos =
            offset
                .checked_add(header_size as u64)
                .ok_or(FormatError::OffsetOverflow {
                    offset,
                    length: header_size as u64,
                })?;
        let mut objects = Vec::new();
        let mut window = DirectoryWindow::new();

        while pos
            .checked_add(2)
            .is_some_and(|index_end| index_end <= collection_end)
        {
            // One request per object: the whole object header where the collection has room
            // for one, and the 2-byte index alone in a tail too short for a header.
            let room_for_header = pos
                .checked_add(object_header_size as u64)
                .is_some_and(|end| end <= collection_end);
            let need = if room_for_header {
                object_header_size
            } else {
                2
            };
            let bytes = window.get(source, pos, need, collection_end)?;
            let object_index = u16::from_le_bytes([bytes[0], bytes[1]]);
            if object_index == 0 {
                break;
            }

            // A short tail could legally hold nothing but that terminator. These
            // are the two checks the walk always made, in the order it made them,
            // so a malformed collection fails with the error it always failed
            // with. Reaching them at all means `bytes` is a whole header.
            let object_header_end =
                pos.checked_add(object_header_size as u64)
                    .ok_or(FormatError::OffsetOverflow {
                        offset: pos,
                        length: object_header_size as u64,
                    })?;
            if object_header_end > collection_end {
                return Err(FormatError::UnexpectedEof {
                    expected: object_header_end.to_usize().unwrap_or(usize::MAX),
                    available: collection_end.to_usize().unwrap_or(usize::MAX),
                });
            }
            let object_size = read_length(bytes, 8, length_size)?;
            let data_address = object_header_end;
            let data_end =
                data_address
                    .checked_add(object_size)
                    .ok_or(FormatError::OffsetOverflow {
                        offset: data_address,
                        length: object_size,
                    })?;
            if data_end > collection_end {
                return Err(FormatError::UnexpectedEof {
                    expected: data_end.to_usize().unwrap_or(usize::MAX),
                    available: collection_end.to_usize().unwrap_or(usize::MAX),
                });
            }

            if keep(object_index) {
                objects.push(GlobalHeapObjectInfo {
                    index: object_index,
                    data_address,
                    size: object_size,
                });
            }

            let padded_size = pad8_u64(object_size)?;
            pos = data_address
                .checked_add(padded_size)
                .ok_or(FormatError::OffsetOverflow {
                    offset: data_address,
                    length: padded_size,
                })?;
        }

        Ok(Self { objects })
    }

    /// Returns the object at `index` in the collection, or `None` if the directory does not hold
    /// it.
    ///
    /// # Performance
    ///
    /// A binary search finds the object in a directory in ascending index order, the order the
    /// encoder writes and the order the C library writes while it only appends to a collection.
    /// The specification allows any order, and where the binary search misses, a linear search
    /// finds the object in a directory of any order.
    pub fn object(&self, index: u16) -> Option<&GlobalHeapObjectInfo> {
        match self.objects.binary_search_by_key(&index, |o| o.index) {
            Ok(pos) => Some(&self.objects[pos]),
            Err(_) => self.objects.iter().find(|object| object.index == index),
        }
    }
}

/// Build one global heap collection holding `objects` (at most
/// [`GLOBAL_HEAP_MAX_OBJECTS`] of them), assigning 1-based object indices in order.
/// Returns the serialized collection bytes.
pub fn encode_global_heap_collection(objects: &[&[u8]]) -> Vec<u8> {
    assert!(
        objects.len() <= GLOBAL_HEAP_MAX_OBJECTS,
        "a collection's 2-byte object index cannot address more than {GLOBAL_HEAP_MAX_OBJECTS} objects"
    );
    // The collection header, sig(4) + ver(1) + reserved(3) + `collection_size`, and each object
    // header, index(2) + reference count(2) + reserved(4) + size, are both this long.
    let header_size = 8 + 8;
    let collection_size = header_size
        + objects
            .iter()
            .map(|obj| header_size + obj.len().next_multiple_of(ALIGNMENT))
            .sum::<usize>()
        + header_size; // free space marker (full object header size)
    let padded_collection = collection_size
        .max(MIN_COLLECTION_SIZE)
        .next_multiple_of(ALIGNMENT);

    let mut buf = Vec::with_capacity(padded_collection);
    // Header
    buf.extend_from_slice(&GCOL_SIGNATURE);
    buf.push(GCOL_VERSION);
    buf.extend_from_slice(&[0u8; 3]); // reserved
    buf.extend_from_slice(&(padded_collection as u64).to_le_bytes());

    // Objects (1-based indices)
    for (i, obj) in objects.iter().enumerate() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the assertion above bounds the object count by `u16::MAX`, so the 1-based \
                      index `i + 1` fits the 2-byte heap object index field"
        )]
        let index = (i + 1) as u16;
        buf.extend_from_slice(&index.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // reference count
        buf.extend_from_slice(&[0u8; 4]); // reserved
        buf.extend_from_slice(&(obj.len() as u64).to_le_bytes());
        buf.extend_from_slice(obj);
        // Pad to 8-byte boundary
        buf.resize(buf.len().next_multiple_of(ALIGNMENT), 0);
    }

    // Free space marker (index 0): the C library uses this size as the total
    // skip distance from the start of the object (including its header), so
    // it must equal the remaining bytes in the collection from this point.
    let free_total_size = padded_collection - buf.len();
    buf.extend_from_slice(&0u16.to_le_bytes()); // index 0
    buf.extend_from_slice(&0u16.to_le_bytes()); // reference count
    buf.extend_from_slice(&[0u8; 4]); // reserved
    buf.extend_from_slice(&(free_total_size as u64).to_le_bytes()); // size

    // Pad collection to full size
    buf.resize(padded_collection, 0);

    buf
}

/// The most objects one global heap collection holds, 65,535.
///
/// The "Heap Object Index" field is 2 bytes wide, and index 0 is the free space. A writer with more
/// objects writes them to several collections, each with its objects at indices from 1. The field
/// is defined in "Global Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
pub const GLOBAL_HEAP_MAX_OBJECTS: usize = u16::MAX as usize;

/// The signature a global heap collection begins with.
const GCOL_SIGNATURE: [u8; 4] = *b"GCOL";

/// The version of a global heap collection, the one version the specification defines.
const GCOL_VERSION: u8 = 1;

// The collection header, each object header and each object's data are padded to a multiple of
// this many bytes, `H5HG_ALIGNMENT` in the C library (`H5HGpkg.h`, HDF5 2.2.0).
const ALIGNMENT: usize = 8;

// The minimum collection size "Global Heap" of the format specification, version 4.0, defines,
// and `H5HG_MINSIZE` in the C library (`H5HGpkg.h`, HDF5 2.2.0).
const MIN_COLLECTION_SIZE: usize = 4096;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::global_heap;
    use test_util::widths::Widths;

    use super::*;

    /// A collection of `(index, reference count, data)` objects.
    fn build_collection(objects: &[(u16, u16, &[u8])], length_size: usize) -> Vec<u8> {
        let objects: Vec<_> = objects
            .iter()
            .map(|&(index, reference_count, data)| global_heap::Object {
                index,
                reference_count,
                data,
            })
            .collect();
        global_heap::collection(&objects, Widths::new(8, length_size))
    }

    #[test]
    fn parse_collection_two_objects() {
        let data = build_collection(&[(1, 1, b"hello"), (2, 1, b"world!!!")], 8);
        let source = data.as_slice();
        let coll = GlobalHeapIndex::parse(source, 0, 8).unwrap();
        assert_eq!(coll.objects.len(), 2);
        assert_eq!(coll.objects[0].index, 1);
        assert_eq!(
            source
                .read_metadata_at(coll.objects[0].data_address, coll.objects[0].size as usize)
                .unwrap(),
            b"hello"
        );
        assert_eq!(coll.objects[1].index, 2);
        assert_eq!(
            source
                .read_metadata_at(coll.objects[1].data_address, coll.objects[1].size as usize)
                .unwrap(),
            b"world!!!"
        );
    }

    #[test]
    fn parse_reads_collection_headers_as_metadata() {
        use core::cell::Cell;

        struct TrackingSource {
            data: Vec<u8>,
            metadata_reads: Cell<usize>,
            raw_reads: Cell<usize>,
        }

        impl MetadataSource for TrackingSource {
            fn len(&self) -> u64 {
                self.data.len() as u64
            }

            fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
                self.raw_reads.set(self.raw_reads.get() + 1);
                self.data.read_at(offset, buf)
            }

            fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
                self.metadata_reads.set(self.metadata_reads.get() + 1);
                self.data.read_metadata_at(offset, len)
            }
        }

        let source = TrackingSource {
            data: build_collection(&[(1, 1, b"hello"), (2, 1, b"world")], 8),
            metadata_reads: Cell::new(0),
            raw_reads: Cell::new(0),
        };

        let coll = GlobalHeapIndex::parse(&source, 0, 8).unwrap();

        assert_eq!(coll.objects.len(), 2);
        assert!(source.metadata_reads.get() > 0);
        assert_eq!(source.raw_reads.get(), 0);
    }

    #[test]
    fn the_directory_walk_reads_by_collection_size_not_by_object_count() {
        use core::cell::Cell;

        struct CountingSource {
            data: Vec<u8>,
            reads: Cell<usize>,
        }

        impl MetadataSource for CountingSource {
            fn len(&self) -> u64 {
                self.data.len() as u64
            }

            fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
                self.data.read_at(offset, buf)
            }

            fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
                self.reads.set(self.reads.get() + 1);
                self.data.read_metadata_at(offset, len)
            }
        }

        // 512 objects of 8 bytes each: 512 * (16 + 8) = 12,288 bytes of
        // directory, three windows' worth.
        const OBJECTS: u16 = 512;
        let payload = [0u8; 8];
        let objects: Vec<(u16, u16, &[u8])> = (1..=OBJECTS).map(|i| (i, 1, &payload[..])).collect();
        let source = CountingSource {
            data: build_collection(&objects, 8),
            reads: Cell::new(0),
        };

        let coll = GlobalHeapIndex::parse(&source, 0, 8).unwrap();
        assert_eq!(coll.objects.len(), OBJECTS as usize);

        // A refill reads `DIRECTORY_LOOKAHEAD` strides of 24 bytes ahead, the headers of about 32
        // objects, and a walk that reads per object takes 512 reads.
        let reads = source.reads.get();
        assert!(
            reads <= OBJECTS as usize / 16,
            "walking {OBJECTS} objects took {reads} reads, which scales with the \
             object count rather than the {} bytes of collection it covers",
            source.data.len()
        );
    }

    #[test]
    fn the_directory_walk_does_not_read_a_window_per_large_object() {
        use core::cell::Cell;

        struct VolumeSource {
            data: Vec<u8>,
            bytes_read: Cell<usize>,
        }

        impl MetadataSource for VolumeSource {
            fn len(&self) -> u64 {
                self.data.len() as u64
            }

            fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
                self.data.read_at(offset, buf)
            }

            fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
                self.bytes_read.set(self.bytes_read.get() + len);
                self.data.read_metadata_at(offset, len)
            }
        }

        // 64 objects of 5,000 bytes, each stride wider than a window.
        const OBJECTS: u16 = 64;
        let payload = vec![0u8; 5000];
        let objects: Vec<(u16, u16, &[u8])> = (1..=OBJECTS).map(|i| (i, 1, &payload[..])).collect();
        let source = VolumeSource {
            data: build_collection(&objects, 8),
            bytes_read: Cell::new(0),
        };

        let coll = GlobalHeapIndex::parse(&source, 0, 8).unwrap();
        assert_eq!(coll.objects.len(), OBJECTS as usize);

        // Reading a window per object takes 64 * 4096 bytes, and the ceiling allows 32 bytes per
        // header and one window.
        let read = source.bytes_read.get();
        let ceiling = DIRECTORY_WINDOW + (OBJECTS as usize + 2) * 32;
        assert!(
            read <= ceiling,
            "walking {OBJECTS} objects of 5,000 bytes read {read} bytes of a \
             {}-byte collection; a window per object rather than a header per \
             object is the failure this bounds",
            source.data.len()
        );
    }

    #[test]
    fn the_directory_walk_does_not_read_a_window_per_alternating_object() {
        use core::cell::Cell;

        struct VolumeSource {
            data: Vec<u8>,
            bytes_read: Cell<usize>,
        }

        impl MetadataSource for VolumeSource {
            fn len(&self) -> u64 {
                self.data.len() as u64
            }

            fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
                self.data.read_at(offset, buf)
            }

            fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
                self.bytes_read.set(self.bytes_read.get() + len);
                self.data.read_metadata_at(offset, len)
            }
        }

        // Alternating 8-byte and 5,000-byte objects, so half the strides fit in a window.
        const OBJECTS: u16 = 256;
        let small = [0u8; 8];
        let large = vec![0u8; 5000];
        let objects: Vec<(u16, u16, &[u8])> = (1..=OBJECTS)
            .map(|i| {
                let payload: &[u8] = if i % 2 == 0 { &large[..] } else { &small[..] };
                (i, 1, payload)
            })
            .collect();
        let source = VolumeSource {
            data: build_collection(&objects, 8),
            bytes_read: Cell::new(0),
        };

        let coll = GlobalHeapIndex::parse(&source, 0, 8).unwrap();
        assert_eq!(coll.objects.len(), OBJECTS as usize);

        // Reading a window after each small object takes about 128 * 4096 bytes, and the ceiling
        // allows 32 bytes per header and one window.
        let read = source.bytes_read.get();
        let ceiling = DIRECTORY_WINDOW + (OBJECTS as usize + 2) * 32;
        assert!(
            read <= ceiling,
            "walking {OBJECTS} alternating objects read {read} bytes of a \
             {}-byte collection, above the {ceiling} a header apiece costs",
            source.data.len()
        );
    }

    #[test]
    fn object_finds_an_object_by_its_index() {
        let data = build_collection(&[(1, 1, b"aaa"), (3, 2, b"bbb")], 8);
        let source = data.as_slice();
        let coll = GlobalHeapIndex::parse(source, 0, 8).unwrap();
        let obj = coll.object(3).unwrap();
        assert_eq!(
            source
                .read_metadata_at(obj.data_address, obj.size as usize)
                .unwrap(),
            b"bbb"
        );
        assert!(coll.object(99).is_none());
    }

    #[rstest]
    #[case::padded_to_the_minimum(&[b"alpha".as_slice(), b"".as_slice()], MIN_COLLECTION_SIZE)]
    #[case::past_the_minimum(&[[7u8; 5000].as_slice()], 5048)]
    fn an_encoded_collection_parses_back_to_its_objects(
        #[case] objects: &[&[u8]],
        #[case] size: usize,
    ) {
        let bytes = encode_global_heap_collection(objects);
        let collection = GlobalHeapIndex::parse(bytes.as_slice(), 0, 8).unwrap();

        assert_eq!(bytes.len(), size);
        assert_eq!(read_length(&bytes, 8, 8), Ok(size as u64));
        let read_back: Vec<(u16, &[u8])> = collection
            .objects
            .iter()
            .map(|object| {
                let start = object.data_address as usize;
                (object.index, &bytes[start..start + object.size as usize])
            })
            .collect();
        let expected: Vec<(u16, &[u8])> = (1..).zip(objects.iter().copied()).collect();
        assert_eq!(read_back, expected);

        let free_space = objects
            .iter()
            .map(|object| 16 + object.len().next_multiple_of(8))
            .sum::<usize>()
            + 16;
        assert_eq!(bytes[free_space..free_space + 8], [0; 8]);
        assert_eq!(
            read_length(&bytes, free_space + 8, 8),
            Ok((size - free_space) as u64)
        );
    }

    #[test]
    fn free_space_terminates_parsing() {
        // Build collection with free space marker immediately
        let mut data = Vec::new();
        data.extend_from_slice(&GCOL_SIGNATURE);
        data.push(1);
        data.extend_from_slice(&[0u8; 3]);
        let size = 8u64 + 8 + 2; // header + `length_size` + free space marker
        data.extend_from_slice(&size.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes()); // free space

        let coll = GlobalHeapIndex::parse(data.as_slice(), 0, 8).unwrap();
        assert_eq!(coll.objects.len(), 0);
    }

    #[test]
    fn invalid_signature_error() {
        let mut data = build_collection(&[(1, 1, b"x")], 8);
        data[0] = b'X'; // corrupt
        let err = GlobalHeapIndex::parse(data.as_slice(), 0, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidGlobalHeapSignature);
    }

    #[test]
    fn invalid_version_error() {
        let mut data = build_collection(&[(1, 1, b"x")], 8);
        data[4] = 2; // wrong version
        let err = GlobalHeapIndex::parse(data.as_slice(), 0, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidGlobalHeapVersion(2));
    }

    #[test]
    fn object_header_cannot_cross_collection_boundary() {
        let mut data = build_collection(&[(1, 1, b"x")], 8);
        let truncated_collection_size = 8u64 + 8 + 2;
        data[8..16].copy_from_slice(&truncated_collection_size.to_le_bytes());

        let err = GlobalHeapIndex::parse(data.as_slice(), 0, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn parse_with_4byte_length() {
        let data = build_collection(&[(1, 1, b"test")], 4);
        let source = data.as_slice();
        let coll = GlobalHeapIndex::parse(source, 0, 4).unwrap();
        assert_eq!(coll.objects.len(), 1);
        let object = &coll.objects[0];
        assert_eq!(
            source
                .read_metadata_at(object.data_address, object.size as usize)
                .unwrap(),
            b"test"
        );
    }
}
