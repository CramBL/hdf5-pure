//! Thread-local test observations of zero generation, chunk preparation, index encoding, and
//! reservation work, including finite append ownership analysis.

use core::cell::Cell;

use hdf5_pure_space::__private::ReservationWork;

/// Zero-generation requests recorded on the current thread.
///
/// Call and byte totals saturate at [`u64::MAX`]. The caller measures issued writes separately.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct ZeroWork {
    /// Recorded zero-generation requests.
    pub(super) calls: u64,
    /// Sum of the byte lengths passed to [`record_zeros`].
    pub(super) bytes: u64,
    /// Largest scratch buffer length supplied to a recorded request.
    pub(super) scratch_peak: u64,
}

std::thread_local! {
    static ZEROS: Cell<ZeroWork> = const { Cell::new(ZeroWork { calls: 0, bytes: 0, scratch_peak: 0 }) };
}

/// Records one zero-generation request and its scratch buffer length.
pub(super) fn record_zeros(bytes: u64, scratch: usize) {
    ZEROS.with(|observed| {
        let mut work = observed.get();
        work.calls = work.calls.saturating_add(1);
        work.bytes = work.bytes.saturating_add(bytes);
        work.scratch_peak = work.scratch_peak.max(scratch as u64);
        observed.set(work);
    });
}

/// Returns the current thread's recorded zero-generation work.
pub(super) fn zero_work() -> ZeroWork {
    ZEROS.with(Cell::get)
}

/// Clears the current thread's zero-generation observations.
pub(super) fn reset_zeros() {
    ZEROS.with(|observed| observed.set(ZeroWork::default()));
}

/// Cumulative chunk preparation, index encoding, and reservation observations on one thread.
///
/// The caller clears cumulative observations through [`reset_creation`]. Successful chunk
/// construction is recorded with the stored length and capacity. Index buffer lengths and
/// capacities are summed at each recorded use, including reuse. Entry into an array encoder
/// can be recorded before placement or encoding fails.
///
/// The caller measures live and peak allocations, buffering, source reads, mirror storage, and
/// issued writes separately. Indexed reservation work excludes scalar searches and other costs
/// described by [`ReservationWork`].
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CreationWork {
    /// Successfully prepared chunks.
    pub(crate) chunks: u64,
    /// Prepared chunks encoded through a filter pipeline.
    pub(crate) filtered_chunks: u64,
    /// Sum of the stored byte lengths of prepared chunks.
    pub(crate) prepared_bytes: u64,
    /// Sum of the byte capacities of prepared chunks at construction.
    pub(crate) prepared_capacity: u64,
    /// Recorded final-index encodings, including array entries before placement.
    pub(crate) index_encodings: u64,
    /// Sum of encoded index buffer lengths, including each recorded block.
    pub(crate) index_bytes: u64,
    /// Sum of index buffer capacities at each recording, including repeated buffer use.
    pub(crate) index_capacity: u64,
    /// Scalar attempts to reserve a preferred allocation from reusable space.
    pub(crate) preferred_probes: u64,
    /// Raw-class capacity queries in PAGE creation planning.
    pub(crate) raw_capacity_scans: u64,
    /// Recorded reservation batches with one request.
    pub(crate) scalar_requests: u64,
    /// Recorded reservation batches that indexed at least one free extent.
    pub(crate) batch_builds: u64,
    /// Sum of request counts supplied to recorded reservation batches.
    pub(crate) requests: u64,
    /// Accumulated indexed work from recorded reservation batches.
    pub(crate) reservations: ReservationWork,
}

std::thread_local! {
    static CREATION: Cell<CreationWork> = Cell::new(CreationWork::default());
}

/// Returns the current thread's cumulative preparation, index, and reservation observations.
pub(crate) fn creation_work() -> CreationWork {
    CREATION.with(Cell::get)
}

/// Clears the current thread's preparation, index, and reservation observations.
pub(crate) fn reset_creation() {
    CREATION.with(|observed| observed.set(CreationWork::default()));
}

/// Records one successful chunk construction and its stored buffer length and capacity.
///
/// The caller passes `true` for `filtered` after encoding through a filter pipeline.
pub(crate) fn record_chunk(bytes: usize, capacity: usize, filtered: bool) {
    CREATION.with(|observed| {
        let mut work = observed.get();
        work.chunks += 1;
        work.filtered_chunks += u64::from(filtered);
        work.prepared_bytes += bytes as u64;
        work.prepared_capacity += capacity as u64;
        observed.set(work);
    });
}

/// Adds one encoded index buffer's byte length and capacity to the cumulative totals.
///
/// Records each encoded block separately, including repeated uses of the same buffer.
pub(crate) fn record_index(bytes: usize, capacity: usize) {
    CREATION.with(|observed| {
        let mut work = observed.get();
        work.index_bytes += bytes as u64;
        work.index_capacity += capacity as u64;
        observed.set(work);
    });
}

/// Records one final-index encoding operation at its caller's observation point.
pub(crate) fn record_index_encoding() {
    CREATION.with(|observed| {
        let mut work = observed.get();
        work.index_encodings += 1;
        observed.set(work);
    });
}

/// Records one scalar preferred-allocation attempt before its reusable-space lookup.
pub(super) fn record_preferred_probe() {
    CREATION.with(|observed| {
        let mut work = observed.get();
        work.preferred_probes += 1;
        observed.set(work);
    });
}

/// Records one raw-capacity query before PAGE creation placement.
pub(super) fn record_raw_capacity_scan() {
    CREATION.with(|observed| {
        let mut work = observed.get();
        work.raw_capacity_scans += 1;
        observed.set(work);
    });
}

/// Adds a reservation batch's request count and indexed work to the current thread's totals.
///
/// The caller passes the full submitted request count, including requests after a failed prefix.
/// The indexed work includes restoration performed when finishing the batch.
pub(super) fn record_reservations(requests: usize, counted: ReservationWork) {
    let ReservationWork {
        indexed_extents,
        lookups,
        removed_extents,
        inserted_extents,
        restored_extents,
    } = counted;
    CREATION.with(|observed| {
        let mut work = observed.get();
        work.requests += requests as u64;
        work.scalar_requests += u64::from(requests == 1);
        work.batch_builds += u64::from(indexed_extents > 0);
        work.reservations.indexed_extents += indexed_extents;
        work.reservations.lookups += lookups;
        work.reservations.removed_extents += removed_extents;
        work.reservations.inserted_extents += inserted_extents;
        work.reservations.restored_extents += restored_extents;
        observed.set(work);
    });
}

std::thread_local! {
    static CHUNK_OWNERSHIP_ANALYSES: Cell<u64> = const { Cell::new(0) };
}

/// Records one finite append ownership analysis on the current thread.
pub(super) fn record_chunk_ownership_analysis() {
    CHUNK_OWNERSHIP_ANALYSES.with(|count| count.set(count.get() + 1));
}

/// Returns the current thread's finite append ownership analysis count.
pub(super) fn chunk_ownership_analyses() -> u64 {
    CHUNK_OWNERSHIP_ANALYSES.with(Cell::get)
}

/// Clears the current thread's finite append ownership analysis count.
pub(super) fn reset_chunk_ownership_analyses() {
    CHUNK_OWNERSHIP_ANALYSES.with(|count| count.set(0));
}

#[cfg(test)]
mod tests;
