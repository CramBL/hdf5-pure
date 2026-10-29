//! Crosschecks transient allocation ownership and resumed native writes.

use core::num::NonZeroUsize;
use std::fs;

use rstest::rstest;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Interval;
use test_util_hdf5::allocation_publication;
use test_util_hdf5::allocation_publication::TailEncoding;

use super::super::Observation;
use super::super::transient;
use super::super::transient::Encoding;
use super::super::transient::Flush;
use super::super::transient::Workload;
use super::super::transient::WritePath;

#[rstest]
#[case::immediate(WritePath::Immediate)]
#[case::borrowed_manual(WritePath::Borrowed(Flush::Manual))]
#[case::borrowed_terminal(WritePath::Borrowed(Flush::Terminal))]
#[case::owned_manual(WritePath::Owned(Flush::Manual))]
#[case::owned_terminal(WritePath::Owned(Flush::Terminal))]
#[case::borrowed_drop(WritePath::Borrowed(Flush::Drop))]
#[case::owned_drop(WritePath::Owned(Flush::Drop))]
fn native_transient_tail_prefixes_conserve_complete_allocations(
    #[case] writer: WritePath,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
    #[values(Encoding::Plain, Encoding::Fletcher32)] encoding: Encoding,
) {
    run(writer, paged, deferred, encoding);
}

#[cfg(feature = "deflate")]
#[rstest]
#[case::compressed(Encoding::Deflate)]
#[case::raw_edges(Encoding::RawEdges)]
fn native_compressed_tails_agree_on_stored_sizes_and_masks(
    #[case] encoding: Encoding,
    #[values(
        WritePath::Immediate,
        WritePath::Borrowed(Flush::Manual),
        WritePath::Owned(Flush::Terminal)
    )]
    writer: WritePath,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
) {
    run(writer, paged, deferred, encoding);
}

fn run(writer: WritePath, paged: bool, deferred: bool, encoding: Encoding) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-tails.h5");
    let native_encoding = match encoding {
        #[cfg(feature = "deflate")]
        Encoding::Deflate => TailEncoding::Deflate,
        Encoding::Fletcher32 => TailEncoding::Fletcher32,
        Encoding::Plain => TailEncoding::Plain,
        #[cfg(feature = "deflate")]
        Encoding::RawEdges => TailEncoding::RawEdges,
    };
    let (address, length) =
        allocation_publication::write_tail_fixture(&path, paged, deferred, native_encoding);
    let hole = Coverage::new([Interval::new(address, length).unwrap()]);
    let original = Observation::read(&fs::read(&path).unwrap());
    assert_eq!(original.index_defined, !deferred);
    assert!(hole.difference(&original.free).is_empty());
    let workload = Workload::record(&path, writer);
    workload.replay(&path, &hole, |point, pure| {
        let native = allocation_publication::Observation::read(&path);
        assert_eq!(native.values, pure.values, "{writer:?}/{encoding:?}/{paged}/{deferred}, prefix {point}");
        assert!(native.chunks.iter().all(|chunk| pure.chunks.contains(chunk)), "{writer:?}/{encoding:?}/{paged}/{deferred}, prefix {point}: native {:?}, reachable {:?}", native.chunks, pure.chunks);
        let mut free = native.free.into_iter().map(|(address, length)| Interval::new(address, length).unwrap()).collect::<Vec<_>>();
        super::super::assert_disjoint(&mut free, "native transient free sections");
        assert_eq!(Coverage::new(free), pure.free, "{writer:?}/{encoding:?}/{paged}/{deferred}, prefix {point}");
    });
    let native = allocation_publication::Observation::read(&path);
    assert_eq!(native.chunks, workload.snapshots.last().unwrap().chunks);
    workload.assert_paths(encoding);
    workload.assert_reclaimed();
    let point = workload.interruption();
    fs::write(&path, workload.prefix(point)).unwrap();
    let before = allocation_publication::Observation::read(&path);
    let retained_raw = workload.snapshots[point].raw_free.clone();
    let address = allocation_publication::append_chunks(&path, NonZeroUsize::new(8).unwrap());
    let after = allocation_publication::Observation::read(&path);
    let pure = Observation::read(&fs::read(&path).unwrap());
    assert_eq!(
        after.values.len(),
        before.values.len() + 8 * super::super::CHUNK
    );
    assert_eq!(pure.values, after.values);
    assert_eq!(pure.chunks, after.chunks);
    assert!(after.chunks.iter().any(|chunk| chunk.1 == address));
    let newly_stored = after
        .chunks
        .iter()
        .filter(|chunk| {
            chunk.0[0]
                >= u64::try_from(
                    before.values.len().div_ceil(super::super::CHUNK) * super::super::CHUNK,
                )
                .unwrap()
        })
        .map(|chunk| Interval::new(chunk.1, chunk.2).unwrap())
        .collect::<Vec<_>>();
    let fresh = Coverage::new(newly_stored.iter().copied()).difference(&retained_raw);
    assert!(
        !fresh.is_empty(),
        "native write used only retained raw tails: {retained_raw:?}"
    );
    if paged {
        assert!(
            newly_stored.iter().map(|range| range.length()).sum::<u64>()
                > retained_raw
                    .intervals()
                    .iter()
                    .map(|range| range.length())
                    .sum::<u64>(),
            "native write did not exceed the retained raw-page capacity"
        );
    }
    let mut sections = after
        .free
        .into_iter()
        .map(|(address, length)| Interval::new(address, length).unwrap())
        .collect::<Vec<_>>();
    super::super::assert_disjoint(&mut sections, "native resumed free sections");
    let free = Coverage::new(sections);
    assert_eq!(pure.free, free);
    pure.inspect(
        "native transient resume",
        point,
        &hole,
        &workload.owned_history(point),
    );
}

#[rstest]
#[case::borrowed(false)]
#[case::owned(true)]
fn native_empty_buffers_and_incomplete_flush_leave_storage_unallocated(
    #[case] owned: bool,
    #[values(false, true)] paged: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-empty.h5");
    let (address, length) =
        allocation_publication::write_tail_fixture(&path, paged, true, TailEncoding::Plain);
    let hole = Coverage::new([Interval::new(address, length).unwrap()]);
    let before = allocation_publication::Observation::read(&path);
    assert!(before.chunks.is_empty());
    transient::assert_empty_operations(&path, owned, &hole);
    let after = allocation_publication::Observation::read(&path);
    assert_eq!(after.free, before.free);
    assert!(after.chunks.is_empty());
    assert!(after.values.is_empty());
}
