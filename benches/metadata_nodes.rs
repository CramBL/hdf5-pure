//! Measures version 1 chunk-node acquisition from borrowed input and a warm metadata cache.
//!
//! Parameters are node-body lengths in bytes. Each timed iteration acquires a fixed population.
//! Fixture construction, cache priming, and decoded-value checks run outside timing.
//!
//! Allocation measurements run after timing with positive controls that allocate a node body.
//! `node_allocation` rows report the source mode, body length, allocation count, allocated bytes,
//! peak bytes, live bytes, and bytes returned through vector metadata reads.

extern crate alloc;

use std::cell::Cell;
use std::hint;
use std::io::Cursor;

use criterion::{BenchmarkId, Criterion, Throughput};
use hdf5_pure_core as error;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{FormatError, MetadataBytes, StoredAddress};
use hdf5_pure_format::__private::{
    self as convert, BTreeV1ChunkNode, BTreeV1ChunkNodeBytes, Narrow,
};
use test_util::allocation;
use test_util::btree_v1;
use test_util::widths::Widths;

use source::SourceMetadata;
use source::{BytesSource, MetadataCacheConfig, MetadataCachingSource, ReadSeekSource, Source};

#[path = "metadata_cache/address.rs"]
mod address;
#[expect(dead_code, reason = "the benchmark only acquires metadata nodes")]
#[allow(
    unfulfilled_lint_expectations,
    reason = "the source module's public API lints do not apply inside a private benchmark module"
)]
#[path = "../src/source.rs"]
mod source;

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

fn main() {
    let fixtures: Vec<_> = [1, 32, 512, 2048].map(Fixture::new).into();
    let mut criterion = Criterion::default().configure_from_args();
    for fixture in &fixtures {
        fixture.time_reads(&mut criterion);
    }
    criterion.final_summary();
    for fixture in &fixtures {
        fixture.measure_reads();
    }
}

/// A population of identical leaf nodes and the sources used to acquire them.
struct Fixture {
    entries: u16,
    node_len: usize,
    offsets: Vec<u64>,
    borrowed: BytesSource<Vec<u8>>,
    cached: MetadataCachingSource<ReadSeekSource<Cursor<Vec<u8>>>>,
}

impl Fixture {
    fn new(entries: u16) -> Self {
        let keys: Vec<_> = (0..=entries)
            .map(|index| btree_v1::chunk_key(4, 0, &[u64::from(index), 0]))
            .collect();
        let children: Vec<_> = (0..entries)
            .map(|index| 0x1000 + u64::from(index) * 4)
            .collect();
        let node = btree_v1::node(
            btree_v1::NodeType::CHUNK,
            0,
            &keys,
            &children,
            Widths::EIGHT,
        );
        let node_len = node.len();
        let image = node.repeat(POPULATION);
        let offsets = (0..POPULATION)
            .map(|index| (index * node_len).to_u64())
            .collect();
        // The cursor contains the complete image and supports every seek the source performs.
        let backend = ReadSeekSource::new(Cursor::new(image.clone())).unwrap();
        let fixture = Self {
            entries,
            node_len,
            offsets,
            borrowed: BytesSource::new(image),
            cached: MetadataCachingSource::new(
                backend,
                MetadataCacheConfig::new(POPULATION * node_len * 2).with_max_entry_bytes(node_len),
            ),
        };
        for offset in &fixture.offsets {
            fixture.check_read(&fixture.borrowed, *offset);
            fixture.check_read(&fixture.cached, *offset);
        }
        fixture.check_cache(0, (2 * POPULATION).to_u64());
        for offset in &fixture.offsets {
            fixture.check_read(&fixture.cached, *offset);
        }
        fixture.check_cache((2 * POPULATION).to_u64(), (2 * POPULATION).to_u64());
        fixture
    }

    fn time_reads(&self, criterion: &mut Criterion) {
        self.time_source(criterion, "borrowed", &self.borrowed);
        self.time_source(criterion, "cached", &self.cached);
    }

    fn time_source(&self, criterion: &mut Criterion, mode: &str, source: &impl Source) {
        for offset in &self.offsets {
            self.check_read(source, *offset);
        }
        self.cached.reset_metadata_cache_stats();
        let metadata = SourceMetadata(source);
        let mut group = criterion.benchmark_group(format!("metadata_node_acquisition/{mode}"));
        group.throughput(Throughput::Elements(POPULATION.to_u64()));
        group.bench_function(
            BenchmarkId::from_parameter(self.node_len - NODE_PREFIX_LEN),
            |bench| {
                bench.iter(|| {
                    for &offset in &self.offsets {
                        // Every fixture offset holds a complete node with eight-byte addresses.
                        hint::black_box(
                            BTreeV1ChunkNodeBytes::read_from_source(
                                &metadata,
                                hint::black_box(offset),
                                8,
                                2,
                            )
                            .unwrap(),
                        );
                    }
                });
            },
        );
        group.finish();
        self.check_cache(0, 0);
    }

    fn measure_reads(&self) {
        self.measure_source("borrowed", &self.borrowed);
        self.measure_source("cached", &self.cached);
    }

    fn measure_source(&self, mode: &str, source: &impl Source) {
        let name = format!("metadata_node_{mode}_{}", self.node_len - NODE_PREFIX_LEN);
        let (control, positive) = allocation::measure(&format!("{name}_control"), || {
            hint::black_box(vec![0u8; self.node_len - NODE_PREFIX_LEN])
        });
        assert!(positive.blocks > 0, "{positive}");
        assert!(
            positive.bytes >= (self.node_len - NODE_PREFIX_LEN).to_u64(),
            "{positive}"
        );
        assert_eq!(control, vec![0u8; self.node_len - NODE_PREFIX_LEN]);
        drop(control);
        let traced = VectorReads {
            inner: source,
            bytes: Cell::new(0),
        };
        let metadata = SourceMetadata(&traced);
        let (holder, measured) = allocation::measure(&name, || {
            // Construction checked the complete node at offset zero before measurement.
            BTreeV1ChunkNodeBytes::read_from_source(&metadata, 0, 8, 2).unwrap()
        });
        self.check_node(holder.node());
        println!(
            "node_allocation,{mode},{},{},{},{},{},{}",
            self.node_len - NODE_PREFIX_LEN,
            measured.blocks,
            measured.bytes,
            measured.peak_bytes,
            measured.live_bytes,
            traced.bytes.get(),
        );
        self.check_cache(0, 0);
    }

    fn check_read(&self, source: &impl Source, offset: u64) {
        let metadata = SourceMetadata(source);
        // The independent fixture builder places a complete node at each supplied offset.
        let holder = BTreeV1ChunkNodeBytes::read_from_source(&metadata, offset, 8, 2).unwrap();
        self.check_node(holder.node());
    }

    fn check_node(&self, node: BTreeV1ChunkNode<'_>) {
        assert_eq!(node.node_level(), 0);
        assert_eq!(node.entries_used(), self.entries);
        assert_eq!(node.stored_len(), self.node_len);
        let mut count = 0;
        for (index, entry) in node.entries().enumerate() {
            // A checked node view contains a complete key and child address for every entry.
            let (key, child) = entry.unwrap();
            assert_eq!(key.chunk_size, 4);
            assert_eq!(key.filter_mask, 0);
            assert_eq!(key.offsets().collect::<Vec<_>>(), [index.to_u64(), 0]);
            assert_eq!(child, StoredAddress::new(0x1000 + index.to_u64() * 4));
            count += 1;
        }
        assert_eq!(count, usize::from(self.entries));
    }

    fn check_cache(&self, expected_hits: u64, expected_misses: u64) {
        // A positive budget enables statistics throughout the cached fixture's lifetime.
        let stats = self.cached.metadata_cache_stats().unwrap();
        assert_eq!(stats.entries(), 2 * POPULATION);
        assert_eq!(stats.bytes(), POPULATION * self.node_len);
        assert!(stats.hits() >= expected_hits);
        assert_eq!(stats.misses(), expected_misses);
        assert_eq!(stats.evictions(), 0);
        assert_eq!(stats.oversize_reads(), 0);
    }
}

/// Counts bytes returned through vector metadata reads.
struct VectorReads<'a, S> {
    inner: &'a S,
    bytes: Cell<usize>,
}

impl<S: Source> Source for VectorReads<'_, S> {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_at(offset, buf)
    }

    fn read_metadata(&self, offset: u64, len: usize) -> Result<MetadataBytes<'_>, FormatError> {
        self.inner.read_metadata(offset, len)
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        let bytes = self.inner.read_metadata_at(offset, len)?;
        self.bytes.set(self.bytes.get() + bytes.len());
        Ok(bytes)
    }
}

/// The length of the signature, node type, level, and entry count, in bytes.
///
/// Defined in "Version 1 B-trees" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
const NODE_PREFIX_LEN: usize = 8;

const POPULATION: usize = 64;
