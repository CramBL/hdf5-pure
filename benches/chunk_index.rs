//! Benchmarks of chunked reads through the Fixed Array, Extensible Array, version 1 B-tree, and
//! Implicit chunk indexes, with what each read allocates.
//!
//! Each fixture is a square `f64` dataset of 64 or 128 chunks to a side, opened twice with no
//! decompressed chunk retained: once with the parsed chunk index retained between reads and once
//! without it. A table of the allocations each read makes follows the timings.
//!
//! ```bash
//! cargo bench --bench chunk_index
//! ```

use std::hint;
use std::time::Duration;

use criterion::BatchSize;
use criterion::BenchmarkId;
use criterion::Criterion;
use hdf5_pure::ChunkCacheConfig;
use hdf5_pure::ChunkIndex;
use hdf5_pure::Dataset;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::MaxExtent;
use test_util::allocation;
use test_util::allocation::Measured;
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

fn main() {
    let fixtures: Vec<_> = [
        CHUNKS_PER_SIDE.map(|chunks_per_side| {
            Fixture::new(ChunkIndex::FixedArray, chunks_per_side, fixed_array_image)
        }),
        CHUNKS_PER_SIDE.map(|chunks_per_side| {
            Fixture::new(
                ChunkIndex::ExtensibleArray,
                chunks_per_side,
                extensible_array_image,
            )
        }),
        CHUNKS_PER_SIDE.map(|chunks_per_side| {
            Fixture::new(ChunkIndex::BTreeV1, chunks_per_side, btree_v1_image)
        }),
        CHUNKS_PER_SIDE.map(|chunks_per_side| {
            Fixture::new(ChunkIndex::Implicit, chunks_per_side, implicit_image)
        }),
    ]
    .into_iter()
    .flatten()
    .collect();

    let mut criterion = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .configure_from_args();
    for fixture in &fixtures {
        fixture.time_reads(&mut criterion);
    }
    criterion.final_summary();

    report_allocations(&fixtures);
}

/// Prints the allocations of each read of each fixture, one row per read, with the allocation
/// count divided by the number of chunks the read selects.
fn report_allocations(fixtures: &[Fixture]) {
    let rows: Vec<_> = fixtures.iter().flat_map(Fixture::measure_reads).collect();
    println!(
        "\n{:<16} {:>6} {:<24} {:>8} {:>11} {:>10} {:>10} {:>10} {:>10}",
        "index",
        "chunks",
        "read",
        "selected",
        "allocations",
        "per chunk",
        "allocated",
        "peak",
        "retained",
    );
    for row in &rows {
        let Row {
            index,
            chunks,
            read,
            selected_chunks,
            measured,
        } = row;
        let per_chunk = measured.blocks as f64 / *selected_chunks as f64;
        println!(
            "{:<16} {chunks:>6} {read:<24} {selected_chunks:>8} {:>11} {per_chunk:>10.2} {:>10} {:>10} {:>10}",
            format!("{index:?}"),
            measured.blocks,
            measured.bytes,
            measured.peak_bytes,
            measured.live_bytes,
        );
    }
}

/// One dataset image, opened with and without the parsed chunk index retained between reads.
struct Fixture {
    index: ChunkIndex,
    chunks_per_side: u64,
    /// The image opened with the parsed chunk index retained between reads.
    cached: File,
    /// The image opened without it.
    uncached: File,
}

impl Fixture {
    /// Builds the image of a dataset `chunks_per_side` chunks to a side and opens it both ways.
    ///
    /// # Panics
    ///
    /// Panics if the dataset does not use `index`, if a whole read returns values other than those
    /// written, or if after a whole read the parsed index is not loaded in the file that retains it
    /// or is loaded in the file that does not.
    fn new(
        index: ChunkIndex,
        chunks_per_side: u64,
        image: impl FnOnce(&[f64], u64) -> Vec<u8>,
    ) -> Self {
        let extent = chunks_per_side * CHUNK_SIDE;
        let values = values(extent);
        let image = image(&values, extent);
        let open = |index_cache| {
            let config = ChunkCacheConfig::new()
                .with_max_bytes(0)
                .with_max_slots(0)
                .with_index_cache(index_cache);
            File::from_bytes_with_options(
                image.clone(),
                FileAccessProperties::new().with_chunk_cache(config),
            )
            .unwrap()
        };
        let fixture = Self {
            index,
            chunks_per_side,
            cached: open(true),
            uncached: open(false),
        };
        let dataset = fixture.cached.dataset(DATASET).unwrap();
        assert_eq!(dataset.chunk_index().unwrap(), Some(index));
        assert_eq!(dataset.read_f64().unwrap(), values);
        assert!(dataset.chunk_cache_stats().index_loaded());
        let dataset = fixture.uncached.dataset(DATASET).unwrap();
        assert_eq!(dataset.read_f64().unwrap(), values);
        assert!(!dataset.chunk_cache_stats().index_loaded());
        fixture
    }

    /// Times each read of this fixture, in a Criterion group named after its chunk index and with
    /// the chunk count as the parameter of each benchmark.
    ///
    /// - `cold_whole`: a whole read on a dataset handle opened for each iteration
    /// - `warm_whole`: a whole read on a handle whose parsed index is loaded
    /// - `uncached_whole`: a whole read on a handle that retains no parsed index
    /// - `warm_windows_NxN` and `uncached_windows_NxN`: [`sweep_windows`] on the same two handles
    fn time_reads(&self, criterion: &mut Criterion) {
        let chunks = self.chunk_count();
        let mut group = criterion.benchmark_group(format!("{:?}", self.index));
        group.bench_function(BenchmarkId::new("cold_whole", chunks), |bench| {
            bench.iter_batched(
                || self.cached.dataset(DATASET).unwrap(),
                |dataset| {
                    let bytes = dataset.read_raw().unwrap();
                    (dataset, bytes)
                },
                BatchSize::SmallInput,
            );
        });
        let warm = self.warm_dataset();
        group.bench_function(BenchmarkId::new("warm_whole", chunks), |bench| {
            bench.iter_with_large_drop(|| warm.read_raw().unwrap());
        });
        let uncached = self.uncached.dataset(DATASET).unwrap();
        group.bench_function(BenchmarkId::new("uncached_whole", chunks), |bench| {
            bench.iter_with_large_drop(|| uncached.read_raw().unwrap());
        });
        for window_side in WINDOW_SIDES_IN_CHUNKS {
            group.bench_function(
                BenchmarkId::new(format!("warm_windows_{window_side}x{window_side}"), chunks),
                |bench| bench.iter(|| sweep_windows(&warm, window_side)),
            );
            group.bench_function(
                BenchmarkId::new(
                    format!("uncached_windows_{window_side}x{window_side}"),
                    chunks,
                ),
                |bench| bench.iter(|| sweep_windows(&uncached, window_side)),
            );
        }
        group.finish();
    }

    /// Measures the allocations of each read that [`Fixture::time_reads`] times, one [`Row`] per
    /// read.
    ///
    /// The cold read is the first read of its handle, and the warm reads after it use the parsed
    /// index it loads.
    fn measure_reads(&self) -> Vec<Row> {
        let chunks = self.chunk_count();
        let row = |read: String, selected_chunks: u64, work: &dyn Fn()| {
            let region = format!("{:?}/{chunks}/{read}", self.index);
            let ((), measured) = allocation::measure(&region, work);
            Row {
                index: self.index,
                chunks,
                read,
                selected_chunks,
                measured,
            }
        };
        let cached = self.cached.dataset(DATASET).unwrap();
        let uncached = self.uncached.dataset(DATASET).unwrap();
        let mut rows = vec![
            row("cold whole".into(), chunks, &|| read_whole(&cached)),
            row("warm whole".into(), chunks, &|| read_whole(&cached)),
            row("uncached whole".into(), chunks, &|| read_whole(&uncached)),
        ];
        for window_side in WINDOW_SIDES_IN_CHUNKS {
            let selected = WINDOWS * window_side * window_side;
            rows.extend([
                row(
                    format!("warm windows {window_side}x{window_side}"),
                    selected,
                    &|| sweep_windows(&cached, window_side),
                ),
                row(
                    format!("uncached windows {window_side}x{window_side}"),
                    selected,
                    &|| sweep_windows(&uncached, window_side),
                ),
            ]);
        }
        rows
    }

    /// Opens the dataset in the file that retains the parsed index and reads it once, so its index
    /// is loaded.
    fn warm_dataset(&self) -> Dataset {
        let dataset = self.cached.dataset(DATASET).unwrap();
        read_whole(&dataset);
        dataset
    }

    fn chunk_count(&self) -> u64 {
        self.chunks_per_side * self.chunks_per_side
    }
}

/// One row of the allocation table: one read of one fixture, and what it allocated.
struct Row {
    index: ChunkIndex,
    /// The number of chunks in the dataset.
    chunks: u64,
    /// The name of the read in the table.
    read: String,
    /// The number of chunks the read selects, the divisor of the per-chunk column.
    selected_chunks: u64,
    measured: Measured,
}

fn read_whole(dataset: &Dataset) {
    hint::black_box(dataset.read_raw().unwrap());
}

/// Reads [`WINDOWS`] square windows of `window_side` chunks to a side along the diagonal of the
/// dataset, each starting where the one before it ends.
fn sweep_windows(dataset: &Dataset, window_side: u64) {
    for window in 0..WINDOWS {
        let start = window * window_side * CHUNK_SIDE;
        let count = window_side * CHUNK_SIDE;
        hint::black_box(
            dataset
                .read_raw_region(&[start, start], &[count, count])
                .unwrap(),
        );
    }
}

/// The elements of an `extent` by `extent` dataset in row-major order, each equal to its own
/// position.
fn values(extent: u64) -> Vec<f64> {
    (0..u32::try_from(extent * extent).unwrap())
        .map(f64::from)
        .collect()
}

/// Writes a dataset whose maximum shape is its shape, which the writer indexes with a Fixed
/// Array.
fn fixed_array_image(values: &[f64], extent: u64) -> Vec<u8> {
    file_builder_image(values, extent, MaxExtent::Fixed(extent))
}

/// Writes a dataset with an unlimited first dimension, which the writer indexes with an
/// Extensible Array.
fn extensible_array_image(values: &[f64], extent: u64) -> Vec<u8> {
    file_builder_image(values, extent, MaxExtent::Unlimited)
}

/// Writes an `extent` by `extent` dataset of `values` in square chunks of [`CHUNK_SIDE`]
/// elements to a side, with `max_rows` as the maximum of its first dimension.
fn file_builder_image(values: &[f64], extent: u64, max_rows: MaxExtent) -> Vec<u8> {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset(DATASET)
        .with_f64_data(values)
        .with_shape(&[extent, extent])
        .with_maxshape(&[max_rows, MaxExtent::Fixed(extent)])
        .with_chunks(&[CHUNK_SIDE, CHUNK_SIDE]);
    builder.finish().unwrap()
}

/// Assembles a file whose dataset indexes its chunks with a version 1 B-tree, with the chunks
/// stored in the row-major order of their origins.
fn btree_v1_image(values: &[f64], extent: u64) -> Vec<u8> {
    let mut image = Image::starting_with(&[0; SUPERBLOCK_SPACE]);
    let origins = chunk_origins(extent);
    let chunk_bytes = u32::try_from(CHUNK_SIDE * CHUNK_SIDE).unwrap() * ELEMENT_SIZE;
    let addresses: Vec<_> = origins
        .iter()
        .map(|&origin| image.append(&chunk_payload(values, extent, origin)) as u64)
        .collect();
    let keys: Vec<_> = origins
        .iter()
        .map(|&[row, column]| btree_v1::chunk_key(chunk_bytes, 0, &[row, column, 0]))
        .chain([btree_v1::chunk_key(0, 0, &[extent, 0, 0])])
        .collect();
    let index = append_chunk_btree(&mut image, &keys, &addresses);
    let chunk_dims = [u32::try_from(CHUNK_SIDE).unwrap(); 2];
    build_dataset(
        image,
        extent,
        &data_layout::chunked(index, &chunk_dims, ELEMENT_SIZE),
    )
}

/// Assembles a file whose dataset indexes its chunks with an Implicit index, with the chunks
/// stored one after another in the row-major order of their origins.
fn implicit_image(values: &[f64], extent: u64) -> Vec<u8> {
    let mut image = Image::starting_with(&[0; SUPERBLOCK_SPACE]);
    let data_address = image.as_bytes().len() as u64;
    for origin in chunk_origins(extent) {
        image.append(&chunk_payload(values, extent, origin));
    }
    let chunk_dims = [u32::try_from(CHUNK_SIDE).unwrap(); 2];
    build_dataset(
        image,
        extent,
        &data_layout::implicit(data_address, &chunk_dims, ELEMENT_SIZE),
    )
}

/// Returns the coordinates of the first element of each chunk of an `extent` by `extent`
/// dataset, in row-major order.
fn chunk_origins(extent: u64) -> Vec<[u64; 2]> {
    (0..extent)
        .step_by(CHUNK_SIDE as usize)
        .flat_map(|row| {
            (0..extent)
                .step_by(CHUNK_SIDE as usize)
                .map(move |column| [row, column])
        })
        .collect()
}

/// Finishes `image` as a file whose root group links an `extent` by `extent` dataset of `f64` as
/// [`DATASET`], with `layout` as its data layout message.
fn build_dataset(image: Image, extent: u64, layout: &[u8]) -> Vec<u8> {
    image.build_with_dataset(
        DATASET,
        [
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
            Message::new(MessageType::DATA_LAYOUT, layout),
        ],
    )
}

/// The little-endian bytes of the chunk whose first element is at `[row, column]`.
fn chunk_payload(values: &[f64], extent: u64, [row, column]: [u64; 2]) -> Vec<u8> {
    (row..row + CHUNK_SIDE)
        .flat_map(|row| {
            let start = usize::try_from(row * extent + column).unwrap();
            &values[start..start + CHUNK_SIDE as usize]
        })
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Appends a version 1 B-tree over the chunks at `chunk_addresses`, one level at a time from the
/// leaves up, and returns the address of its root.
///
/// `keys` holds the key of each chunk and then the key that closes the last one.
fn append_chunk_btree(image: &mut Image, keys: &[Vec<u8>], chunk_addresses: &[u64]) -> u64 {
    let mut nodes: Vec<(u64, usize)> = chunk_addresses.iter().copied().zip(0..).collect();
    let mut level = 0;
    loop {
        nodes = append_btree_level(image, level, keys, &nodes);
        if let [(root, _)] = nodes[..] {
            return root;
        }
        level += 1;
    }
}

/// Appends the nodes of one level of the tree, up to [`ENTRIES_PER_NODE`] children to a node,
/// and returns the address of each node with the position in `keys` of the first chunk below it.
///
/// Each of `children` is the address of a child and the position in `keys` of the first chunk
/// below it. A node stores the address of the sibling that follows it, so the address of every
/// node at the level is computed before the first one is appended.
fn append_btree_level(
    image: &mut Image,
    level: u8,
    keys: &[Vec<u8>],
    children: &[(u64, usize)],
) -> Vec<(u64, usize)> {
    let groups: Vec<_> = children.chunks(ENTRIES_PER_NODE).collect();
    let node_keys: Vec<Vec<_>> = groups
        .iter()
        .enumerate()
        .map(|(position, group)| {
            let end = groups
                .get(position + 1)
                .map_or(keys.len() - 1, |next| next[0].1);
            group
                .iter()
                .map(|&(_, first_chunk)| keys[first_chunk].clone())
                .chain([keys[end].clone()])
                .collect()
        })
        .collect();
    let child_addresses: Vec<Vec<_>> = groups
        .iter()
        .map(|group| group.iter().map(|&(address, _)| address).collect())
        .collect();
    let encode = |position: usize, left_sibling, right_sibling| {
        btree_v1::node_with_siblings(
            btree_v1::NodeType::CHUNK,
            level,
            &node_keys[position],
            &child_addresses[position],
            left_sibling,
            right_sibling,
            Widths::EIGHT,
        )
    };
    let addresses: Vec<_> = (0..groups.len())
        .scan(image.as_bytes().len() as u64, |next, position| {
            let address = *next;
            *next += encode(position, None, None).len() as u64;
            Some(address)
        })
        .collect();
    groups
        .iter()
        .enumerate()
        .map(|(position, group)| {
            let left_sibling = position.checked_sub(1).map(|left| addresses[left]);
            let right_sibling = addresses.get(position + 1).copied();
            let address = image.append(&encode(position, left_sibling, right_sibling)) as u64;
            assert_eq!(address, addresses[position]);
            (address, group[0].1)
        })
        .collect()
}

/// The link name of the dataset in every fixture.
const DATASET: &str = "data";

/// The side of a square chunk, in elements.
const CHUNK_SIDE: u64 = 4;

/// The sides of the fixture datasets, in chunks.
const CHUNKS_PER_SIDE: [u64; 2] = [64, 128];

/// The number of windows [`sweep_windows`] reads in one sweep.
const WINDOWS: u64 = 32;

const WINDOW_SIDES_IN_CHUNKS: [u64; 2] = [1, 2];

/// The most entries libhdf5 puts in a node of a chunk B-tree under its default creation
/// properties, twice `HDF5_BTREE_CHUNK_IK_DEF` (`H5Fprivate.h`, HDF5 2.2.0).
const ENTRIES_PER_NODE: usize = 64;

/// The size of an `f64` element in bytes.
const ELEMENT_SIZE: u32 = 8;

/// The number of zero bytes the version 1 B-tree and Implicit fixtures reserve at offset 0 for the
/// superblock that [`Image::build_with_dataset`] writes.
const SUPERBLOCK_SPACE: usize = 64;
