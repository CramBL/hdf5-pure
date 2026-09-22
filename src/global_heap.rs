//! HDF5 Global Heap collection parsing.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::bytes::read_length;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::Source;

/// Magic signature for global heap collections.
const GCOL_SIGNATURE: [u8; 4] = *b"GCOL";

/// Metadata index for a global heap collection.
///
/// This stores object locations rather than copying every object payload. VL
/// readers can parse a shared collection once and fetch only the referenced
/// object bytes.
#[derive(Debug, Clone)]
pub struct GlobalHeapIndex {
    /// Object locations within this collection.
    pub objects: Vec<GlobalHeapObjectInfo>,
}

/// Location and size of one object in a global heap collection.
#[derive(Debug, Clone)]
pub struct GlobalHeapObjectInfo {
    /// Object index (1-based; 0 is the free space marker).
    pub index: u16,
    /// Absolute address of the object payload.
    pub data_address: u64,
    /// Object payload size in bytes.
    pub size: u64,
}

fn pad8_u64(x: u64) -> Result<u64, FormatError> {
    x.checked_add(7)
        .map(|value| value & !7)
        .ok_or(FormatError::OffsetOverflow {
            offset: x,
            length: 7,
        })
}

/// The most of a collection the directory walk holds at a time.
///
/// One page: enough that a collection of many small objects pays one read per few
/// hundred of them. It is a ceiling and not the size of every read — see
/// [`DirectoryWindow::get`], which reads only what it needs once the objects are
/// too far apart for a window to span two headers.
const DIRECTORY_WINDOW: usize = 4096;

/// How many recent strides decide a refill's span. One object larger than a
/// window silences the read-ahead for this many objects, which is what keeps a
/// collection of mixed sizes at one header-sized read per object instead of one
/// window per object.
const DIRECTORY_STRIDE_HISTORY: usize = 8;

/// How many headers a refill reaches for, at the widest recent spacing. Bounds
/// what a wrong guess costs: a refill reads this many strides, not a flat
/// [`DIRECTORY_WINDOW`] regardless of how far apart the headers are.
const DIRECTORY_LOOKAHEAD: u64 = 32;

/// A bounded sliding view of one collection's bytes.
///
/// The directory walk has to pass *every* object in a collection — each object's
/// size chains the position of the next — so a read per object is a read per
/// object of the whole collection however few of them the caller wanted. A
/// 32,768-object collection cost 65,536 reads and as many allocations to resolve
/// a 256-element window (issue #228); through this window it costs one per few
/// hundred objects.
///
/// Refills go through [`Source::read_metadata_at`], so whatever metadata cache
/// the source has still serves them and still holds the same bytes — what
/// changes is how many times it is asked.
struct DirectoryWindow {
    /// Absolute offset the buffer begins at.
    start: u64,
    bytes: Vec<u8>,
    /// Where the previous request was, so the next refill can tell how far apart
    /// this collection's object headers are. `None` until the first one.
    last: Option<u64>,
    /// The last [`DIRECTORY_STRIDE_HISTORY`] gaps between requested positions,
    /// most recent last. Zero means "not yet observed", which a real stride
    /// never is, since the walk's position strictly increases.
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

    /// The `need` bytes at `pos`, refilling from `source` when this window does
    /// not already hold them.
    ///
    /// # How much a refill reads
    ///
    /// Reading a whole window ahead pays only when the next header lands inside
    /// it. A collection of objects larger than a page has one header per window,
    /// and reading 4 KiB to take 16 bytes out of it is *worse* than the two small
    /// reads this replaced — measured at 225x the bytes for 100 objects of 5 KB,
    /// which is not a corner case: the reference C library gives an object larger
    /// than its collection one of its own, so a dataset of large variable-length
    /// values is exactly this shape.
    ///
    /// So the span is decided by how far apart recent headers have been. Two
    /// things matter about *which* recent headers, and the first version of this
    /// got both wrong by looking only at the last stride:
    ///
    /// * The widest of the last [`DIRECTORY_STRIDE_HISTORY`] strides decides,
    ///   not the most recent one. A stride predicts the *next* gap, and one
    ///   object's size says nothing about the next object's; in a collection
    ///   whose sizes alternate, the stride into a header is smallest exactly
    ///   when the stride out of it is largest, so reading on that signal reads a
    ///   whole window and discards it at every transition. Taking the widest
    ///   recent stride reads ahead only where objects have been *consistently*
    ///   close, which is the only shape read-ahead pays on.
    /// * The span is [`DIRECTORY_LOOKAHEAD`] strides, not a flat window, so what
    ///   a wrong guess costs stays proportional to the spacing that justified it
    ///   rather than being 4 KiB regardless.
    ///
    /// With no stride recorded yet the widest is zero and the span is `need`, so
    /// the first refill reads nothing extra.
    ///
    /// # Bounds
    ///
    /// A refill reads no more than `limit`, the end of the collection being
    /// walked, leaves — so a window stays inside the structure it views, and
    /// cannot reach past the end of the source either, since the caller has
    /// already established that `limit` is within it. `need` is a floor on the
    /// span, so a caller asking for more bytes than the collection holds reads
    /// past `limit` rather than indexing out of bounds; both call sites below
    /// establish that `need` fits before asking, which is what keeps that
    /// unreachable.
    fn get<'a, S: Source + ?Sized>(
        &'a mut self,
        source: &S,
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
                // The widest recent gap, so that one large object among small
                // ones stops the read-ahead rather than being averaged away.
                let widest = self.strides.iter().copied().max().unwrap_or(0);
                let ahead = if widest.saturating_add(need as u64) > DIRECTORY_WINDOW as u64 {
                    // Headers are further apart than a window, so a window could
                    // not hold a second one: read this header and nothing else.
                    need
                } else {
                    // Cover the next `DIRECTORY_LOOKAHEAD` headers at the widest
                    // spacing seen. Zero (nothing observed yet) lands on `need`.
                    widest
                        .saturating_mul(DIRECTORY_LOOKAHEAD)
                        .saturating_add(need as u64)
                        .min(DIRECTORY_WINDOW as u64)
                        .to_usize()
                        .unwrap_or(need)
                        .max(need)
                };
                // Clamped to one window before the narrowing, so the conversion
                // cannot lose anything on a 32-bit target and the fallback is
                // unreachable — it is spelled out rather than asserted because
                // the clamp above is what makes it so.
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
    /// Parse collection metadata from a random-access source without copying
    /// object payloads.
    pub fn parse<S: Source + ?Sized>(
        source: &S,
        offset: u64,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        Self::parse_filtered(source, offset, length_size, |_| true)
    }

    /// [`parse`](Self::parse), retaining only the objects `keep` accepts.
    ///
    /// The walk still visits every object header — each object's size chains
    /// the position of the next — but the directory holds just the accepted
    /// entries, so a caller resolving a few objects of a large collection
    /// (e.g. a row window of a variable-length string dataset) is not charged
    /// the whole collection's directory.
    pub(crate) fn parse_filtered<S: Source + ?Sized>(
        source: &S,
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
        if version != 1 {
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
            // One request per object. The index is the first two bytes of the
            // object header, so the header is read whole wherever the collection
            // has room for one, and the two-byte terminator probe is what is left
            // for a tail too short to hold another header. Asking twice at the
            // same position would refill twice on the narrow path above.
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
            // with — and reaching them at all means `bytes` is a whole header.
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

    /// Get object metadata by its collection-local index.
    ///
    /// A collection holds up to 65,535 objects and a variable-length read
    /// resolves one lookup per element, so the common case must not be a linear
    /// scan. Writers lay objects out in ascending index order (this crate's
    /// certainly, and the reference C library's while a collection is only
    /// appended to), which makes the directory sorted and the lookup a binary
    /// search; the format does not guarantee that ordering, so an out-of-order
    /// collection falls back to a scan rather than reporting a present object
    /// as missing.
    pub fn get_object(&self, index: u16) -> Option<&GlobalHeapObjectInfo> {
        match self.objects.binary_search_by_key(&index, |o| o.index) {
            Ok(pos) => Some(&self.objects[pos]),
            Err(_) => self.objects.iter().find(|object| object.index == index),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::BytesSource;
    use test_util::global_heap;
    use test_util::widths::Widths;

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
        let source = BytesSource::new(&data);
        let coll = GlobalHeapIndex::parse(&source, 0, 8).unwrap();
        assert_eq!(coll.objects.len(), 2);
        assert_eq!(coll.objects[0].index, 1);
        assert_eq!(
            source
                .read_exact_at(coll.objects[0].data_address, coll.objects[0].size as usize)
                .unwrap(),
            b"hello"
        );
        assert_eq!(coll.objects[1].index, 2);
        assert_eq!(
            source
                .read_exact_at(coll.objects[1].data_address, coll.objects[1].size as usize)
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

        impl Source for TrackingSource {
            fn len(&self) -> u64 {
                self.data.len() as u64
            }

            fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
                self.raw_reads.set(self.raw_reads.get() + 1);
                BytesSource::new(&self.data).read_at(offset, buf)
            }

            fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
                self.metadata_reads.set(self.metadata_reads.get() + 1);
                BytesSource::new(&self.data).read_exact_at(offset, len)
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

    /// The walk reads through a window, so what it costs follows the collection's
    /// *size* and not its object count — the thing that made resolving a 256-row
    /// window of a 32,768-object collection cost 65,536 reads (issue #228).
    ///
    /// Stated as a bound on reads rather than on time: every one of them is an
    /// allocation, and on a source with a metadata cache it is also a lookup, an
    /// insert and an eviction, none of which a timing test would name.
    #[test]
    fn the_directory_walk_reads_by_collection_size_not_by_object_count() {
        use core::cell::Cell;

        struct CountingSource {
            data: Vec<u8>,
            reads: Cell<usize>,
        }

        impl Source for CountingSource {
            fn len(&self) -> u64 {
                self.data.len() as u64
            }

            fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
                BytesSource::new(&self.data).read_at(offset, buf)
            }

            fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
                self.reads.set(self.reads.get() + 1);
                BytesSource::new(&self.data).read_exact_at(offset, len)
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

        // One read must serve many objects. The exact ratio follows
        // `DIRECTORY_LOOKAHEAD` and the object size — these are 8-byte objects,
        // the smallest a collection holds, so their stride buys the least
        // read-ahead of any shape. A walk that reads per object takes 512 reads
        // or twice that; the point of the bound is the gap between the two, not
        // its exact value.
        let reads = source.reads.get();
        assert!(
            reads <= OBJECTS as usize / 16,
            "walking {OBJECTS} objects took {reads} reads, which scales with the \
             object count rather than the {} bytes of collection it covers",
            source.data.len()
        );
    }

    /// The other side of the window's bargain, and the one it got wrong first.
    ///
    /// Reading a window ahead pays only when the next header lands inside it. A
    /// collection of objects larger than a page has one header per window, and a
    /// fixed-size window read 4 KiB to take 16 bytes out of it — 225x the bytes
    /// the two small reads it replaced took, on a shape the reference C library
    /// produces routinely (an object larger than a collection gets one of its
    /// own). So the walk must read *no more than the old one did* here, even as
    /// it reads far less often for small objects.
    #[test]
    fn the_directory_walk_does_not_read_a_window_per_large_object() {
        use core::cell::Cell;

        struct VolumeSource {
            data: Vec<u8>,
            bytes_read: Cell<usize>,
        }

        impl Source for VolumeSource {
            fn len(&self) -> u64 {
                self.data.len() as u64
            }

            fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
                BytesSource::new(&self.data).read_at(offset, buf)
            }

            fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
                self.bytes_read.set(self.bytes_read.get() + len);
                BytesSource::new(&self.data).read_exact_at(offset, len)
            }
        }

        // 64 objects of 5,000 bytes: every stride is wider than a window, so
        // every header is a miss.
        const OBJECTS: u16 = 64;
        let payload = vec![0u8; 5000];
        let objects: Vec<(u16, u16, &[u8])> = (1..=OBJECTS).map(|i| (i, 1, &payload[..])).collect();
        let source = VolumeSource {
            data: build_collection(&objects, 8),
            bytes_read: Cell::new(0),
        };

        let coll = GlobalHeapIndex::parse(&source, 0, 8).unwrap();
        assert_eq!(coll.objects.len(), OBJECTS as usize);

        // One 16-byte header apiece, the collection header, and one window's
        // worth of slack for the first refill, which has no stride to judge by.
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

    /// The shape that defeats a read-ahead judged by the *last* stride: object
    /// sizes that alternate, so the gap into each header is smallest exactly
    /// when the gap out of it is largest.
    ///
    /// A rule that read a window whenever the previous object was small read
    /// 4 KiB and used 16 bytes of it at every transition — 8.3 MB of a 10 MB
    /// file to resolve a 16-row window, 54x what the per-object reads it
    /// replaced cost. Deciding on the *widest* recent stride instead reads
    /// ahead only where objects have been consistently close, so this shape
    /// costs a header apiece.
    ///
    /// Bounded against the per-object read it must not lose to, rather than
    /// against a measured figure, so the rule is what holds and not one
    /// fixture's answer.
    #[test]
    fn the_directory_walk_does_not_read_a_window_per_alternating_object() {
        use core::cell::Cell;

        struct VolumeSource {
            data: Vec<u8>,
            bytes_read: Cell<usize>,
        }

        impl Source for VolumeSource {
            fn len(&self) -> u64 {
                self.data.len() as u64
            }

            fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
                BytesSource::new(&self.data).read_at(offset, buf)
            }

            fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
                self.bytes_read.set(self.bytes_read.get() + len);
                BytesSource::new(&self.data).read_exact_at(offset, len)
            }
        }

        // Alternating 8-byte and 5,000-byte objects: half the strides fit in a
        // window and half do not, and they interleave, so no single previous
        // stride predicts the next.
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

        // The same allowance the uniform large-object case gets: a header
        // apiece, the collection header, and one window of slack for the first
        // refill, which has no stride to judge by. A window per transition is
        // roughly 128 * 4096 and fails this by two orders of magnitude.
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
    fn get_object_by_index() {
        let data = build_collection(&[(1, 1, b"aaa"), (3, 2, b"bbb")], 8);
        let source = BytesSource::new(&data);
        let coll = GlobalHeapIndex::parse(&source, 0, 8).unwrap();
        let obj = coll.get_object(3).unwrap();
        assert_eq!(
            source
                .read_exact_at(obj.data_address, obj.size as usize)
                .unwrap(),
            b"bbb"
        );
        assert!(coll.get_object(99).is_none());
    }

    #[test]
    fn free_space_terminates_parsing() {
        // Build collection with free space marker immediately
        let mut data = Vec::new();
        data.extend_from_slice(&GCOL_SIGNATURE);
        data.push(1);
        data.extend_from_slice(&[0u8; 3]);
        let size = 8u64 + 8 + 2; // header + length_size + free space marker
        data.extend_from_slice(&size.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes()); // free space

        let coll = GlobalHeapIndex::parse(&BytesSource::new(&data), 0, 8).unwrap();
        assert_eq!(coll.objects.len(), 0);
    }

    #[test]
    fn invalid_signature_error() {
        let mut data = build_collection(&[(1, 1, b"x")], 8);
        data[0] = b'X'; // corrupt
        let err = GlobalHeapIndex::parse(&BytesSource::new(&data), 0, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidGlobalHeapSignature);
    }

    #[test]
    fn invalid_version_error() {
        let mut data = build_collection(&[(1, 1, b"x")], 8);
        data[4] = 2; // wrong version
        let err = GlobalHeapIndex::parse(&BytesSource::new(&data), 0, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidGlobalHeapVersion(2));
    }

    #[test]
    fn object_header_cannot_cross_collection_boundary() {
        let mut data = build_collection(&[(1, 1, b"x")], 8);
        let truncated_collection_size = 8u64 + 8 + 2;
        data[8..16].copy_from_slice(&truncated_collection_size.to_le_bytes());

        let err = GlobalHeapIndex::parse(&BytesSource::new(&data), 0, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn parse_with_4byte_length() {
        let data = build_collection(&[(1, 1, b"test")], 4);
        let source = BytesSource::new(&data);
        let coll = GlobalHeapIndex::parse(&source, 0, 4).unwrap();
        assert_eq!(coll.objects.len(), 1);
        let object = &coll.objects[0];
        assert_eq!(
            source
                .read_exact_at(object.data_address, object.size as usize)
                .unwrap(),
            b"test"
        );
    }
}
