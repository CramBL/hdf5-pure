//! Benchmarks whole reads of dense and sparse chunked datasets.
//!
//! Run with `cargo bench --locked --bench dense_chunk_reads`. Each fixture is an unfiltered,
//! square [`f64`] dataset with a version 1 B-tree chunk index and 16 or 64 logical chunk slots.
//! Outputs range from 2 KiB to 32 MiB. The workloads use the default zero fill value, an explicit
//! nonzero fill value, or that nonzero fill value with an interior hole.
//!
//! Cold reads use a new dataset handle, warm reads retain a parsed index, and uncached reads
//! disable index retention. Decoded chunks are never retained. Every mode batches one read per
//! iteration. Timing excludes handle setup, output destruction, and handle destruction.
//!
//! Throughput uses logical output bytes, including sparse holes. The allocation table records
//! read allocations, peak and retained bytes, and zeroed allocation requests. Payload and scatter
//! byte counts are calculated from fixture geometry, excluding fill pattern tiling and hardware
//! cache effects. Measuring hardware memory traffic requires separate instrumentation.

use std::error::Error;
use std::hint;
use std::time::Duration;

use criterion::BatchSize;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use hdf5_pure::ChunkCacheConfig;
use hdf5_pure::Dataset;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FillTime;
use hdf5_pure_format::__private::FillValueMessage;
use heapscope::Snapshot;
use test_util::allocation;
use test_util::btree_v1;
use test_util::data_layout;
use test_util::dataspace;
use test_util::datatype;
use test_util::image::Image;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::widths::Widths;

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

fn main() -> Result<(), Box<dyn Error>> {
    let fixtures = WORKLOADS
        .iter()
        .flat_map(|&workload| SHAPES.map(|(grid, chunk)| Fixture::new(workload, grid, chunk)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut criterion = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .configure_from_args();
    for fixture in &fixtures {
        fixture.time_reads(&mut criterion);
    }
    criterion.final_summary();
    allocation::measure("initialize_profiler", || ());
    println!(
        "\nworkload,grid,chunk_side,read,output_bytes,allocated_chunks,payload_read_bytes,scatter_write_bytes,zeroed_request_bytes,allocations,allocated_bytes,peak_bytes,retained_bytes"
    );
    for fixture in &fixtures {
        fixture.report_reads();
    }
    println!(
        "Payload and scatter byte counts are calculated from fixture geometry. Zeroed bytes count allocator requests on this serial read, including scratch buffers. These counts exclude fill tiling and hardware cache effects."
    );
    Ok(())
}

/// A file image opened from memory with parsed index retention enabled and disabled.
struct Fixture {
    workload: Workload,
    chunks_per_side: u64,
    chunk_side: u64,
    /// The file opened with parsed index retention enabled.
    cached: File,
    /// The file opened with parsed index retention disabled.
    uncached: File,
    output_bytes: u64,
    /// The total bytes stored in allocated chunk payloads.
    payload_bytes: u64,
    allocated_chunks: usize,
}

impl Fixture {
    fn new(
        workload: Workload,
        chunks_per_side: u64,
        chunk_side: u64,
    ) -> Result<Self, Box<dyn Error>> {
        let extent = chunks_per_side * chunk_side;
        let mut image = Image::starting_with(&[0; 64]);
        let mut addresses = Vec::new();
        let mut keys = Vec::new();
        let mut expected = vec![workload.fill.unwrap_or(0.0); usize::try_from(extent * extent)?];
        let chunk_bytes = u32::try_from(chunk_side * chunk_side * ELEMENT_BYTES)?;
        for row in 0..chunks_per_side {
            for column in 0..chunks_per_side {
                if workload.sparse
                    && (chunks_per_side / 4..3 * chunks_per_side / 4).contains(&row)
                    && (chunks_per_side / 4..3 * chunks_per_side / 4).contains(&column)
                {
                    continue;
                }
                let mut payload = Vec::with_capacity(usize::try_from(chunk_bytes)?);
                for y in row * chunk_side..(row + 1) * chunk_side {
                    for x in column * chunk_side..(column + 1) * chunk_side {
                        let value = f64::from(u32::try_from(y * extent + x)?);
                        expected[usize::try_from(y * extent + x)?] = value;
                        payload.extend_from_slice(&value.to_le_bytes());
                    }
                }
                addresses.push(u64::try_from(image.append(&payload))?);
                keys.push(btree_v1::chunk_key(
                    chunk_bytes,
                    0,
                    &[row * chunk_side, column * chunk_side, 0],
                ));
            }
        }
        keys.push(btree_v1::chunk_key(0, 0, &[extent, 0, 0]));
        let index = u64::try_from(image.append(&btree_v1::node(
            btree_v1::NodeType::CHUNK,
            0,
            &keys,
            &addresses,
            Widths::EIGHT,
        )))?;
        let mut messages = vec![
            Message::new(
                MessageType::DATASPACE,
                &dataspace::v2(
                    2,
                    dataspace::Flags::NONE,
                    dataspace::Kind::SIMPLE,
                    &[extent, extent],
                    None,
                ),
            ),
            Message::new(MessageType::DATATYPE, &datatype::f64_le()),
            Message::new(
                MessageType::DATA_LAYOUT,
                &data_layout::chunked(index, &[u32::try_from(chunk_side)?; 2], 8),
            ),
        ];
        if let Some(fill) = workload.fill {
            // The fill value is eight bytes, so its length fits the message's u32 field.
            let fill = FillValueMessage::new(FillTime::IfSet, Some(fill.to_le_bytes().to_vec()))
                .serialize()
                .unwrap();
            messages.push(Message::new(MessageType::FILL_VALUE, &fill));
        }
        let image = image.build_with_dataset(DATASET, messages);
        let open = |index_cache| {
            File::from_bytes_with_options(
                image.clone(),
                FileAccessProperties::new().with_chunk_cache(
                    ChunkCacheConfig::new()
                        .with_max_bytes(0)
                        .with_max_slots(0)
                        .with_index_cache(index_cache),
                ),
            )
        };
        let cached = open(true)?;
        let uncached = open(false)?;
        let fixture = Self {
            workload,
            chunks_per_side,
            chunk_side,
            cached,
            uncached,
            output_bytes: extent * extent * ELEMENT_BYTES,
            payload_bytes: u64::try_from(addresses.len())? * u64::from(chunk_bytes),
            allocated_chunks: addresses.len(),
        };
        fixture.verify(&expected);
        Ok(fixture)
    }

    fn verify(&self, expected: &[f64]) {
        for (file, retains_index) in [(&self.cached, true), (&self.uncached, false)] {
            let dataset = dataset(file);
            assert!(!dataset.chunk_cache_stats().index_loaded());
            // An incorrect fixture aborts the benchmark before any timing begins.
            assert_eq!(dataset.read_f64().unwrap(), expected);
            assert_eq!(dataset.chunk_cache_stats().index_loaded(), retains_index);
            assert_eq!(dataset.chunk_cache_stats().cached_chunks(), 0);
        }
    }

    /// Measures whole reads with identical batching for every index cache mode.
    fn time_reads(&self, criterion: &mut Criterion) {
        let Self {
            workload,
            chunks_per_side,
            chunk_side,
            cached,
            uncached,
            output_bytes,
            payload_bytes: _,
            allocated_chunks: _,
        } = self;
        let mut group = criterion.benchmark_group(workload.name);
        let shape = format!("grid_{chunks_per_side}/chunk_{chunk_side}");
        group.throughput(Throughput::Bytes(*output_bytes));
        group.bench_function(BenchmarkId::new("cold_index", &shape), |bench| {
            bench.iter_batched(
                || dataset(cached),
                |dataset| {
                    let bytes = read(&dataset);
                    (dataset, bytes)
                },
                BatchSize::PerIteration,
            );
        });
        let warm = dataset(cached);
        drop(read(&warm));
        group.bench_function(BenchmarkId::new("warm_index", &shape), |bench| {
            bench.iter_batched(|| (), |()| read(&warm), BatchSize::PerIteration);
        });
        let uncached = dataset(uncached);
        group.bench_function(BenchmarkId::new("uncached_index", &shape), |bench| {
            bench.iter_batched(|| (), |()| read(&uncached), BatchSize::PerIteration);
        });
        group.finish();
    }

    /// Prints allocation measurements and logical byte counts for each index cache mode.
    fn report_reads(&self) {
        let Self {
            workload,
            chunks_per_side,
            chunk_side,
            cached,
            uncached,
            output_bytes,
            payload_bytes,
            allocated_chunks,
        } = self;
        let cold = dataset(cached);
        let uncached = dataset(uncached);
        for (mode, dataset) in [
            ("cold_index", &cold),
            ("warm_index", &cold),
            ("uncached_index", &uncached),
        ] {
            let region = format!("{}/{chunks_per_side}/{chunk_side}/{mode}", workload.name);
            let before = Snapshot::capture().shapes.zeroed_bytes;
            let (bytes, measured) = allocation::measure(&region, || read(dataset));
            let zeroed_bytes = Snapshot::capture().shapes.zeroed_bytes - before;
            assert_eq!(u64::try_from(bytes.len()), Ok(*output_bytes));
            assert!(measured.bytes >= *output_bytes);
            assert!(measured.live_bytes >= *output_bytes);
            println!(
                "{},{chunks_per_side},{chunk_side},{mode},{output_bytes},{allocated_chunks},{payload_bytes},{payload_bytes},{zeroed_bytes},{},{},{},{}",
                workload.name,
                measured.blocks,
                measured.bytes,
                measured.peak_bytes,
                measured.live_bytes,
            );
        }
    }
}

/// Fill settings and chunk allocation for a benchmark fixture.
#[derive(Clone, Copy)]
struct Workload {
    name: &'static str,
    fill: Option<f64>,
    sparse: bool,
}

fn dataset(file: &File) -> Dataset {
    // Both immutable file images contain the dataset's root link.
    file.dataset(DATASET).unwrap()
}

fn read(dataset: &Dataset) -> Vec<u8> {
    // Fixture construction verifies a complete read before measurement.
    hint::black_box(dataset.read_raw().unwrap())
}

const DATASET: &str = "values";
const SHAPES: [(u64, u64); 4] = [(4, 4), (8, 4), (4, 256), (8, 256)];
const ELEMENT_BYTES: u64 = 8;
const WORKLOADS: [Workload; 3] = [
    Workload {
        name: "dense_zero_fill",
        fill: None,
        sparse: false,
    },
    Workload {
        name: "dense_nonzero_fill",
        fill: Some(7.0),
        sparse: false,
    },
    Workload {
        name: "sparse_nonzero_fill",
        fill: Some(7.0),
        sparse: true,
    },
];
