//! Reading a run of adjacent chunks in one call, for the streaming readers.
//!
//! # Why this exists
//!
//! The streaming chunked readers fetch one chunk per [`Source::read_exact_at`],
//! which is one `seek` plus one `read` each. That is right for the
//! megabyte-sized chunks a file written in one pass carries, and wrong for a
//! file written as the data arrived: a recorder that appends a row at a time
//! produces one chunk per row, so a recording of 73 datasets over 280 rows
//! holds 20,440 chunks of a few dozen bytes. Reading one chunk at a time cost
//! about four times what the same read costs from a buffered
//! [`crate::File::open`], and the whole difference was read *granularity* — the
//! file was read exactly once, in 32-byte pieces.
//!
//! Those chunks are adjacent on disk. libhdf5 allocates a chunk when its chunk
//! cache flushes rather than when the write is issued, so even a writer filling
//! 73 datasets a row at a time lays each dataset's chunks down in one run, and
//! this crate's writer does the same. A reader that asks for the run in one
//! call gets the same bytes in 73 reads instead of 20,440, which brings the
//! streaming read within about 15% of the buffered one.
//!
//! # What it does not spend
//!
//! **Memory.** A span stops at [`MAX_SPAN_BYTES`], so what a reader holds
//! beyond the caller's own buffers is bounded by that however large the dataset
//! is. `open_streaming` exists so that a file larger than memory still opens,
//! and this must not spend that. The one span that may exceed the cap is a
//! single chunk larger than it, which is read alone — exactly the read the
//! direct path would have issued.
//!
//! **Read volume.** Only chunks that are exactly adjacent merge, so a span
//! covers nothing but the chunks it was planned over. Bridging a gap between
//! two nearby chunks would trade read count for read volume; on the layouts
//! this targets there is no gap to bridge. It follows that the caller must plan
//! over the chunks it will actually fetch, not over every chunk of the dataset:
//! a chunk the caller will skip — outside a row window, or already in the chunk
//! cache — leaves a hole that a span would cover and read for nothing.
//!
//! # What the caller owes it
//!
//! Chunks must be *asked for* in address order. A reader holds one span, so a
//! walk that alternates between two spans re-reads a whole span at each
//! alternation; the chunk index yields index order and a cached index yields no
//! order at all, so neither is safe to walk unsorted. Serving out of order
//! stays correct — [`chunk_bytes`](ChunkSpanReader::chunk_bytes) re-reads
//! whatever span it needs — but it gives up the read count that is the point.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::Source;

/// Largest span read in one call, and so the most a [`ChunkSpanReader`] holds.
///
/// The value is a ceiling, not a target. A chunk-per-row recording's run is one
/// dataset long — 8,960 bytes on the file above — so any cap past a few tens of
/// kilobytes captures the same win there; what the cap decides is the layouts
/// where a run really is long, a dataset of 4 KiB chunks written in one pass,
/// and there it is the whole memory bound.
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
