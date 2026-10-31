//! Measures metadata-cache hit throughput across entry counts and payload sizes.
//!
//! The payload-size group uses the cached source path with a fixed population. Cache setup and
//! value checks are outside measured work.

extern crate alloc;

use std::hint;
use std::io::Cursor;
use std::sync::Mutex;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hdf5_pure_core as error;
use hdf5_pure_core::MetadataBytes;
use hdf5_pure_format::__private::{self as convert, Narrow};

use source::{
    MetadataCacheConfig, MetadataCachingSource, MetadataReadCache, ReadSeekSource, Source,
};

#[expect(
    dead_code,
    reason = "the benchmark does not reconcile or inspect opened image addresses"
)]
#[allow(
    unused_imports,
    reason = "the shared address module also exports types used by the reader"
)]
#[path = "../src/address.rs"]
mod address;
#[expect(dead_code, reason = "the benchmark only reads the metadata cache")]
#[allow(
    unfulfilled_lint_expectations,
    reason = "the source module's public API lints do not apply inside a private benchmark module"
)]
#[path = "../src/source/backend.rs"]
mod source;

fn metadata_cache_hits(criterion: &mut Criterion) {
    const ENTRY_LEN: usize = 64;
    let mut group = criterion.benchmark_group("metadata_cache_hits");
    for entries in [1_024, 16_384] {
        let config = MetadataCacheConfig::new(entries * ENTRY_LEN * 2);
        let cache = Mutex::new(MetadataReadCache::new());
        let offsets: Vec<_> = (0..entries)
            .map(|index| u64::try_from(index * ENTRY_LEN).unwrap())
            .collect();
        for &offset in &offsets {
            MetadataReadCache::read_through(&cache, config, offset, ENTRY_LEN, || {
                MetadataBytes::from_vec(offset, vec![7; ENTRY_LEN])
            })
            .unwrap();
        }
        let mut cache = cache.into_inner().unwrap();
        assert_eq!(cache.stats().entries(), entries);
        for &offset in &offsets {
            assert_eq!(
                cache.get(offset, ENTRY_LEN).unwrap().as_ref(),
                &[7; ENTRY_LEN]
            );
        }
        group.throughput(Throughput::Elements(u64::try_from(entries).unwrap()));
        group.bench_function(BenchmarkId::from_parameter(entries), |bench| {
            bench.iter(|| {
                for &offset in &offsets {
                    hint::black_box(cache.get(hint::black_box(offset), ENTRY_LEN).unwrap());
                }
            });
        });
    }
    group.finish();
}

fn metadata_cache_payload_hits(criterion: &mut Criterion) {
    const ENTRIES: usize = 64;
    let mut group = criterion.benchmark_group("metadata_cache_payload_hits");
    group.throughput(Throughput::Elements(ENTRIES.to_u64()));
    for entry_len in [64, 1_024, 16_384, 65_536] {
        let config =
            MetadataCacheConfig::new(ENTRIES * entry_len * 2).with_max_entry_bytes(entry_len);
        // A cursor over the complete byte vector supports every seek used by the constructor.
        let backend = ReadSeekSource::new(Cursor::new(vec![7; ENTRIES * entry_len])).unwrap();
        let source = MetadataCachingSource::new(backend, config);
        let offsets: Vec<_> = (0..ENTRIES)
            .map(|index| (index * entry_len).to_u64())
            .collect();
        let expected = vec![7; entry_len];
        for &offset in &offsets {
            // Each offset selects one complete entry in the cursor's vector.
            assert_eq!(
                source.read_metadata(offset, entry_len).unwrap().as_ref(),
                expected
            );
        }
        // The positive byte budget enables statistics for this source.
        let stats = source.metadata_cache_stats().unwrap();
        assert_eq!(stats.entries(), ENTRIES);
        assert_eq!(stats.bytes(), ENTRIES * entry_len);
        assert_eq!(stats.misses(), ENTRIES.to_u64());
        assert_eq!(stats.hits(), 0);
        assert_eq!(stats.evictions(), 0);
        assert_eq!(stats.oversize_reads(), 0);
        for &offset in &offsets {
            // Every requested entry was admitted during the first pass.
            assert_eq!(
                source.read_metadata(offset, entry_len).unwrap().as_ref(),
                expected
            );
        }
        // The source's enabled cache retains all entries after both passes.
        let stats = source.metadata_cache_stats().unwrap();
        assert_eq!(stats.entries(), ENTRIES);
        assert_eq!(stats.bytes(), ENTRIES * entry_len);
        assert_eq!(stats.hits(), ENTRIES.to_u64());
        assert_eq!(stats.misses(), ENTRIES.to_u64());
        source.reset_metadata_cache_stats();
        group.bench_function(BenchmarkId::from_parameter(entry_len), |bench| {
            bench.iter(|| {
                for &offset in &offsets {
                    // The configured cache retains every requested entry throughout timing.
                    hint::black_box(
                        source
                            .read_metadata(hint::black_box(offset), entry_len)
                            .unwrap(),
                    );
                }
            });
        });
        // Resetting counters preserves the enabled cache and its retained entries.
        let stats = source.metadata_cache_stats().unwrap();
        assert_eq!(stats.entries(), ENTRIES);
        assert_eq!(stats.bytes(), ENTRIES * entry_len);
        assert_eq!(stats.misses(), 0);
        assert_eq!(stats.evictions(), 0);
        assert_eq!(stats.oversize_reads(), 0);
    }
    group.finish();
}

criterion_group!(benches, metadata_cache_hits, metadata_cache_payload_hits);
criterion_main!(benches);
