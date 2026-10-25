//! Coalesced reads of adjacent chunks, for the streaming chunked readers.
//!
//! A [`ChunkSpanReader`] merges the chunks that are adjacent in the file into spans, reads each
//! span in one [`Source::read_at`] call, and serves each chunk's bytes out of the span it holds.
//! A span is at most [`MAX_SPAN_BYTES`] long unless it is a single chunk larger than that, and the
//! reader holds one span at a time.
//!
//! The caller plans the spans over the chunks it fetches, not over every chunk of the dataset. A
//! read skips a chunk outside its region or in the chunk cache, and the reader reads the whole of
//! a span, including a skipped chunk in it. The caller requests the chunks in ascending order of
//! address, the order of a [`ParsedChunkIndex`](crate::parsed_chunk_index::ParsedChunkIndex).
//! Out of order, the reader returns the same bytes and reads a span again each time the caller
//! returns to it.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::address::StoredAddress;
#[cfg(test)]
use crate::address::StoredAddressExt;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::Source;

/// The longest span of more than one chunk that a [`ChunkSpanReader`] reads in one call.
///
/// The limit bounds the memory of a read of a dataset whose chunks lie end to end, such as one of
/// 4 KiB chunks written in one pass.
pub(crate) const MAX_SPAN_BYTES: u64 = 256 * 1024;

/// One contiguous byte range of the file, covering one or more chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    start: u64,
    /// Exclusive.
    end: u64,
}

impl Span {
    fn covers(&self, start: u64, end: u64) -> bool {
        start >= self.start && end <= self.end
    }

    fn len(&self) -> u64 {
        self.end - self.start
    }
}

/// Serves a planned set of chunks out of coalesced reads.
///
/// Build it with the chunks one read will fetch (see the module docs on why
/// that is not the same as the dataset's chunk list), then ask for each chunk's
/// bytes with [`chunk_bytes`](Self::chunk_bytes) in address order. A chunk in
/// the span already held is served from the buffer; otherwise its whole span is
/// read first.
pub(crate) struct ChunkSpanReader {
    /// Sorted by `start`, and non-overlapping by construction.
    spans: Vec<Span>,
    /// The span `buf` currently holds, or `None` when `buf` is stale.
    held: Option<Span>,
    buf: Vec<u8>,
    /// Holds a chunk that belongs to no span, so serving one does not evict the
    /// span its neighbours are being served from.
    scratch: Vec<u8>,
}

impl ChunkSpanReader {
    /// Plan spans over `chunks`, given as `(address, size)`, or `None` when
    /// coalescing them would not pay.
    ///
    /// `None` means the caller should read each chunk directly, which is the
    /// answer whenever no two chunks are adjacent: every span would be a single
    /// chunk, and serving it through the buffer would only add a copy. A file
    /// whose writer had its chunk cache disabled scatters its chunks that way
    /// and pays nothing for this.
    pub(crate) fn new<I, N>(chunks: I) -> Option<Self>
    where
        I: IntoIterator<Item = (StoredAddress, N)>,
        N: Into<u64>,
    {
        let mut ranges: Vec<Span> = chunks
            .into_iter()
            .map(|(address, size)| {
                let size = size.into();
                Span {
                    start: address.get(),
                    // A crafted address near `u64::MAX` saturates here. The
                    // read itself is bounds-checked by the source.
                    end: address.get().saturating_add(size),
                }
            })
            .collect();
        if ranges.len() < 2 {
            return None;
        }
        ranges.sort_unstable_by_key(|s| s.start);

        let mut spans: Vec<Span> = Vec::new();
        for range in ranges.iter().copied() {
            match spans.last_mut() {
                // Adjacent to the span being built (`start == last.end`), or
                // overlapping it in a file whose chunks overlap, and still
                // within the budget measured from that span's start. The bound
                // is exactly `last.end`: one byte more would bridge a
                // single-byte gap, and fetch a byte belonging to no chunk.
                Some(last)
                    if range.start <= last.end
                        && range.end.saturating_sub(last.start) <= MAX_SPAN_BYTES =>
                {
                    last.end = last.end.max(range.end);
                }
                _ => spans.push(range),
            }
        }

        // Nothing merged, so every span is one chunk and there is nothing here
        // the direct path does not already do.
        if spans.len() == ranges.len() {
            return None;
        }
        Some(Self {
            spans,
            held: None,
            buf: Vec::new(),
            scratch: Vec::new(),
        })
    }

    /// The `len` bytes of the chunk at `address`, reading its span first if
    /// that is not the span already held.
    ///
    /// `source` is framed at the file's base address, the frame the chunk index
    /// stores its addresses in.
    pub(crate) fn chunk_bytes<S: Source + ?Sized>(
        &mut self,
        source: &S,
        address: StoredAddress,
        len: usize,
    ) -> Result<&[u8], FormatError> {
        let address = address.get();
        let end = address.saturating_add(len as u64);

        if !matches!(self.held, Some(span) if span.covers(address, end)) {
            let Some(span) = self.span_containing(address, end) else {
                // Not a chunk this reader was planned over. Serve it directly
                // and leave the held span alone.
                self.scratch = source.read_exact_at(address, len)?;
                return Ok(&self.scratch);
            };
            // Reuse the buffer rather than replacing it, so peak stays one span
            // rather than the two a fresh allocation would briefly hold. The
            // held span is cleared first: a failed read leaves `buf` holding
            // bytes that belong to no span, and nothing may serve out of it.
            self.held = None;
            self.buf.resize(span.len().to_usize()?, 0);
            source.read_at(span.start, &mut self.buf)?;
            self.held = Some(span);
        }

        let span = self.held.expect("held above, or set just above");
        let lo = (address - span.start).to_usize()?;
        lo.checked_add(len)
            .and_then(|hi| self.buf.get(lo..hi))
            .ok_or(FormatError::UnexpectedEof {
                expected: lo.saturating_add(len),
                available: self.buf.len(),
            })
    }

    /// The span covering `[start, end)`, if this reader was planned over a
    /// chunk there.
    fn span_containing(&self, start: u64, end: u64) -> Option<Span> {
        let idx = match self.spans.binary_search_by_key(&start, |s| s.start) {
            Ok(i) => i,
            Err(0) => return None,
            Err(i) => i - 1,
        };
        self.spans
            .get(idx)
            .copied()
            .filter(|s| s.covers(start, end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::BytesSource;
    use core::cell::Cell;

    /// A source that records every read it serves.
    struct CountingSource {
        bytes: Vec<u8>,
        reads: Cell<usize>,
        bytes_read: Cell<usize>,
        largest: Cell<usize>,
    }

    impl CountingSource {
        fn new(len: usize) -> Self {
            Self {
                bytes: (0..len).map(|i| i as u8).collect(),
                reads: Cell::new(0),
                bytes_read: Cell::new(0),
                largest: Cell::new(0),
            }
        }
    }

    impl Source for CountingSource {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            self.reads.set(self.reads.get() + 1);
            self.bytes_read.set(self.bytes_read.get() + buf.len());
            self.largest.set(self.largest.get().max(buf.len()));
            BytesSource::new(&self.bytes).read_at(offset, buf)
        }
    }

    /// `count` chunks of `size` bytes laid end to end from `start`.
    fn run(start: u64, size: u32, count: usize) -> Vec<(StoredAddress, u32)> {
        (0..count)
            .map(|i| (StoredAddress::new(start + i as u64 * u64::from(size)), size))
            .collect()
    }

    #[test]
    fn one_read_serves_a_whole_run_of_chunks() {
        let chunks = run(16, 8, 100);
        let source = CountingSource::new(4096);
        let mut reader = ChunkSpanReader::new(chunks.clone()).expect("a run of chunks coalesces");

        for &(address, _) in &chunks {
            let bytes = reader.chunk_bytes(&source, address, 8).unwrap();
            let expected: Vec<u8> = (address.get()..address.get() + 8)
                .map(|b| b as u8)
                .collect();
            assert_eq!(bytes, &expected[..], "chunk at {address:?}");
        }

        // One read for the whole run, and not a byte beyond it.
        assert_eq!(source.reads.get(), 1);
        assert_eq!(source.bytes_read.get(), 800);
    }

    /// The budget is what keeps a long run from being read whole, so the test
    /// that guards it asserts the span count the budget implies rather than
    /// comparing against the constant — an assertion written in terms of
    /// `MAX_SPAN_BYTES` holds for every value of it, including the ones that
    /// give up the bound.
    #[test]
    fn a_long_run_is_split_at_the_budget() {
        // 40 chunks of 8 KiB is 320 KiB, so a 256 KiB budget takes the first
        // 32 and leaves 8 for a second span.
        let chunks = run(0, 8192, 40);
        let source = CountingSource::new(40 * 8192);
        let mut reader = ChunkSpanReader::new(chunks.clone()).expect("adjacent chunks coalesce");

        for &(address, size) in &chunks {
            let bytes = reader.chunk_bytes(&source, address, size as usize).unwrap();
            assert_eq!(bytes.len(), 8192);
            assert_eq!(bytes[0], address.get() as u8);
        }

        assert_eq!(source.reads.get(), 2, "320 KiB of chunks is two spans");
        assert_eq!(source.largest.get(), 32 * 8192, "the first span is full");
        // Still exactly the chunks' bytes, spread over the two spans.
        assert_eq!(source.bytes_read.get(), 40 * 8192);
    }

    #[test]
    fn chunks_that_are_not_adjacent_are_left_to_the_direct_path() {
        let scattered: Vec<(StoredAddress, u32)> =
            (0..16).map(|i| (StoredAddress::new(i * 4096), 8)).collect();
        assert!(ChunkSpanReader::new(scattered).is_none());
    }

    /// The gap that separates a span from the next one is one byte, and a
    /// reader that bridged it would fetch a byte belonging to no chunk — the
    /// one thing this is not allowed to do. A wider gap does not test the
    /// boundary; an off-by-one in the adjacency test survives it.
    #[test]
    fn a_one_byte_gap_is_not_bridged() {
        let gapped: Vec<(StoredAddress, u32)> =
            (0..16).map(|i| (StoredAddress::new(i * 33), 32)).collect();
        assert!(
            ChunkSpanReader::new(gapped).is_none(),
            "chunks one byte apart are not adjacent"
        );

        // And in a layout that does coalesce, the gap still bounds the span:
        // four adjacent chunks, one byte, four more.
        let mut mixed = run(0, 32, 4);
        mixed.extend(run(4 * 32 + 1, 32, 4));
        let source = CountingSource::new(1024);
        let mut reader = ChunkSpanReader::new(mixed.clone()).expect("the runs coalesce");
        for &(address, size) in &mixed {
            reader.chunk_bytes(&source, address, size as usize).unwrap();
        }
        assert_eq!(source.reads.get(), 2, "the gap splits the run");
        assert_eq!(
            source.bytes_read.get(),
            8 * 32,
            "the byte between the runs was not read"
        );
    }

    #[test]
    fn a_single_chunk_is_left_to_the_direct_path() {
        assert!(ChunkSpanReader::new(run(0, 64, 1)).is_none());
    }

    #[test]
    fn a_partly_adjacent_layout_still_coalesces_its_runs() {
        // Two runs of four, far apart: two spans, so two reads for eight chunks.
        let mut chunks = run(0, 32, 4);
        chunks.extend(run(100_000, 32, 4));
        let source = CountingSource::new(200_000);
        let mut reader = ChunkSpanReader::new(chunks.clone()).expect("two runs coalesce");

        for &(address, _) in &chunks {
            reader.chunk_bytes(&source, address, 32).unwrap();
        }
        assert_eq!(source.reads.get(), 2);
        assert_eq!(source.bytes_read.get(), 8 * 32);
    }

    #[test]
    fn a_chunk_outside_every_span_is_read_directly() {
        let chunks = run(0, 32, 8);
        let source = CountingSource::new(200_000);
        let mut reader = ChunkSpanReader::new(chunks).expect("adjacent chunks coalesce");

        reader
            .chunk_bytes(&source, StoredAddress::new(0), 32)
            .unwrap();
        assert_eq!(source.reads.get(), 1);

        // A chunk this reader was not planned over still reads correctly...
        let stray = reader
            .chunk_bytes(&source, StoredAddress::new(100_000), 4)
            .unwrap()
            .to_vec();
        assert_eq!(stray, vec![160u8, 161, 162, 163]);
        assert_eq!(source.reads.get(), 2);

        // ...and does not cost the held span, which still serves its chunks.
        reader
            .chunk_bytes(&source, StoredAddress::new(224), 32)
            .unwrap();
        assert_eq!(source.reads.get(), 2);
    }

    #[test]
    fn a_chunk_larger_than_the_budget_is_read_on_its_own() {
        // One over-budget chunk, then a run of small ones. The big chunk cannot
        // join a span, so it is read alone — the same read it would have been
        // without coalescing — and does not drag its neighbours over budget.
        let big = MAX_SPAN_BYTES + 40 * 1024;
        let big_usize = big.to_usize().unwrap();
        let mut chunks = vec![(StoredAddress::new(0), u32::try_from(big).unwrap())];
        chunks.extend(run(big, 64, 8));

        let source = CountingSource::new(big_usize + 8 * 64);
        let mut reader = ChunkSpanReader::new(chunks.clone()).expect("the small chunks coalesce");

        assert_eq!(
            reader
                .chunk_bytes(&source, StoredAddress::new(0), big_usize)
                .unwrap()
                .len(),
            big_usize
        );
        assert_eq!(source.largest.get(), big_usize, "the big chunk read alone");

        for &(address, _) in &chunks[1..] {
            reader.chunk_bytes(&source, address, 64).unwrap();
        }
        // The big chunk, then one span over the eight small ones.
        assert_eq!(source.reads.get(), 2);
        assert_eq!(source.bytes_read.get(), big_usize + 8 * 64);
    }

    /// Out-of-order requests must still return the right bytes. They cost
    /// reads, which is why the callers sort; correctness must not depend on
    /// their having done so.
    #[test]
    fn chunks_are_served_whatever_order_they_are_asked_for() {
        let chunks = run(64, 16, 8);
        let source = CountingSource::new(1024);
        let mut reader = ChunkSpanReader::new(chunks.clone()).expect("adjacent chunks coalesce");

        for &(address, _) in chunks.iter().rev() {
            let bytes = reader.chunk_bytes(&source, address, 16).unwrap();
            assert_eq!(bytes[0], address.get() as u8);
        }
        assert_eq!(source.reads.get(), 1, "one span serves both directions");
    }

    /// The span buffer is reused, so it must be cut to each span's own length:
    /// a buffer only ever grown would make a short span read the length of the
    /// longest one before it, which is bytes belonging to no chunk.
    #[test]
    fn a_short_span_reads_its_own_length_after_a_longer_one() {
        let mut chunks = run(0, 64, 8); // 512 bytes
        chunks.extend(run(100_000, 8, 2)); // 16 bytes, far away
        let source = CountingSource::new(200_000);
        let mut reader = ChunkSpanReader::new(chunks).expect("two runs coalesce");

        reader
            .chunk_bytes(&source, StoredAddress::new(0), 64)
            .unwrap();
        let before = source.bytes_read.get();
        let short = reader
            .chunk_bytes(&source, StoredAddress::new(100_000), 8)
            .unwrap();
        assert_eq!(short, &[160u8, 161, 162, 163, 164, 165, 166, 167]);
        assert_eq!(source.bytes_read.get() - before, 16, "the short span's own");
    }
}
