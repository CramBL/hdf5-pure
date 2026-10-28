//! Measures metadata-cache hit throughput at different cache sizes.
//!
//! Cache setup and value checks are outside measured work.

extern crate alloc;

use std::hint;
use std::sync::Mutex;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hdf5_pure_core as error;
use hdf5_pure_core::MetadataBytes;
use hdf5_pure_format::__private as convert;

use source::{MetadataCacheConfig, MetadataReadCache};

#[path = "metadata_cache/address.rs"]
mod address;
#[expect(dead_code, reason = "the benchmark only reads the metadata cache")]
#[allow(
    unfulfilled_lint_expectations,
    reason = "the source module's public API lints do not apply inside a private benchmark module"
)]
#[path = "../src/source.rs"]
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

criterion_group!(benches, metadata_cache_hits);
criterion_main!(benches);
