//! Benchmarks for numeric conversion, dataset I/O, and file-space operations.
//!
//! Primitive groups measure decoding without dataset I/O, encoding without file serialization,
//! and buffered appends below a chunk boundary. Decode controls measure allocation and byte-copy
//! costs. Dataset groups include layout assembly and filters. Inputs and fixtures are prepared
//! outside measured work.
//!
//! Run with:
//!
//! ```bash
//! cargo bench --bench hot_paths
//! ```
//!
//! Compare times for the same workload and environment.

use std::time::{Duration, Instant};

use criterion::BenchmarkId;
use criterion::{Criterion, criterion_group, criterion_main};
use hdf5_pure::DeflateLevel;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::H5Element;
use hdf5_pure::{File, FileBuilder, MaxExtent};
use test_util_hdf5::file_space;

/// Build a file image from a configuring closure.
fn build(configure: impl FnOnce(&mut FileBuilder)) -> Vec<u8> {
    let mut b = FileBuilder::new();
    configure(&mut b);
    b.finish().expect("serialize file")
}

// ---------------------------------------------------------------------------
// Contiguous typed decode (isolates the data_read byte->typed conversion)
// ---------------------------------------------------------------------------

fn bench_contiguous_decode(c: &mut Criterion) {
    let n: usize = 1 << 20; // ~1M elements

    let f64_bytes = build(|b| {
        let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        b.create_dataset("d").with_f64_data(&data);
    });
    let i32_bytes = build(|b| {
        let data: Vec<i32> = (0..n)
            .map(|i| (i as i32).wrapping_mul(2654435761u32 as i32))
            .collect();
        b.create_dataset("d").with_i32_data(&data);
    });
    let u16_bytes = build(|b| {
        let data: Vec<u16> = (0..n).map(|i| (i & 0xFFFF) as u16).collect();
        b.create_dataset("d").with_u16_data(&data);
    });

    let f64_file = File::from_bytes(f64_bytes).unwrap();
    let i32_file = File::from_bytes(i32_bytes).unwrap();
    let u16_file = File::from_bytes(u16_bytes).unwrap();

    let mut g = c.benchmark_group("contiguous_decode_1M");
    g.bench_function("read_f64", |b| {
        b.iter(|| std::hint::black_box(f64_file.dataset("d").unwrap().read_f64().unwrap()))
    });
    g.bench_function("read_i32", |b| {
        b.iter(|| std::hint::black_box(i32_file.dataset("d").unwrap().read_i32().unwrap()))
    });
    g.bench_function("read_u16", |b| {
        b.iter(|| std::hint::black_box(u16_file.dataset("d").unwrap().read_u16().unwrap()))
    });
    // Cross-type coercion: decode an i32-stored dataset to f64 (FixedPoint path).
    g.bench_function("read_i32_as_f64", |b| {
        b.iter(|| std::hint::black_box(i32_file.dataset("d").unwrap().read_f64().unwrap()))
    });
    g.finish();
}

fn bench_primitive_decode(c: &mut Criterion) {
    let count = 1 << 20;
    let mut builder = FileBuilder::new();
    builder.create_dataset("i32").with_i32_data(&[0]);
    builder.create_dataset("f64").with_f64_data(&[0.0]);
    builder.create_dataset("u16").with_u16_data(&[0]);
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let i32_type = file.dataset("i32").unwrap().datatype().unwrap();
    let f64_type = file.dataset("f64").unwrap().datatype().unwrap();
    let u16_type = file.dataset("u16").unwrap().datatype().unwrap();
    let i32_raw: Vec<_> = (0..count as i32).flat_map(i32::to_le_bytes).collect();
    let f64_raw: Vec<_> = (0..count)
        .flat_map(|value| (value as f64 * 0.5).to_le_bytes())
        .collect();
    let u16_raw: Vec<_> = (0..count)
        .flat_map(|value| (value as u16).to_le_bytes())
        .collect();
    let mut group = c.benchmark_group("primitive_decode_1M");
    group.bench_function("i32", |bench| {
        bench.iter(|| {
            std::hint::black_box(
                i32::convert_raw(std::hint::black_box(&i32_raw), &i32_type).unwrap(),
            )
        })
    });
    group.bench_function("f64", |bench| {
        bench.iter(|| {
            std::hint::black_box(
                f64::convert_raw(std::hint::black_box(&f64_raw), &f64_type).unwrap(),
            )
        })
    });
    group.bench_function("u16", |bench| {
        bench.iter(|| {
            std::hint::black_box(
                u16::convert_raw(std::hint::black_box(&u16_raw), &u16_type).unwrap(),
            )
        })
    });
    group.bench_function("i32_as_f64", |bench| {
        bench.iter(|| {
            std::hint::black_box(
                f64::convert_raw(std::hint::black_box(&i32_raw), &i32_type).unwrap(),
            )
        })
    });
    group.bench_function("copy_4MiB", |bench| {
        bench.iter(|| std::hint::black_box(std::hint::black_box(&i32_raw).to_vec()))
    });
    group.bench_function("allocate_4MiB", |bench| {
        bench.iter(|| std::hint::black_box(Vec::<i32>::with_capacity(std::hint::black_box(count))))
    });
    group.finish();
}

fn bench_primitive_encode(c: &mut Criterion) {
    let count: usize = 1 << 20;
    let integers: Vec<_> = (0..count as i32).collect();
    let floats: Vec<_> = (0..count).map(|value| value as f64 * 0.5).collect();
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("values");
    let mut group = c.benchmark_group("primitive_encode_1M");
    group.bench_function("i32", |bench| {
        bench.iter(|| {
            dataset.with_i32_data(std::hint::black_box(&integers));
            std::hint::black_box(&*dataset);
        })
    });
    group.bench_function("f64", |bench| {
        bench.iter(|| {
            dataset.with_f64_data(std::hint::black_box(&floats));
            std::hint::black_box(&*dataset);
        })
    });
    group.finish();
}

fn bench_primitive_append(c: &mut Criterion) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("primitive-append.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_i32_data(&[])
        .with_chunks(&[8_192])
        .with_maxshape(&[MaxExtent::Unlimited]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let mut dataset = file.dataset("values").unwrap();
    let values = [17i32; 16];
    c.bench_function("primitive_append_256x16_i32", |bench| {
        bench.iter_custom(|iterations| {
            let mut elapsed = Duration::ZERO;
            for _ in 0..iterations {
                let mut appender = dataset.buffered_appender().unwrap();
                appender.append(&values).unwrap();
                let start = Instant::now();
                for _ in 0..256 {
                    appender.append(std::hint::black_box(&values)).unwrap();
                }
                elapsed += start.elapsed();
                std::hint::black_box(appender.discard());
            }
            elapsed
        })
    });
}

// ---------------------------------------------------------------------------
// Chunked assembly (isolates copy_chunk_to_output scatter into the output buf)
// ---------------------------------------------------------------------------

fn bench_chunked_assembly(c: &mut Criterion) {
    let rows = 1024u64;
    let cols = 1024u64;
    let data: Vec<f64> = (0..(rows * cols)).map(|i| i as f64).collect();

    // Uncompressed chunked: read time is dominated by chunk assembly + decode.
    let plain = build(|b| {
        b.create_dataset("d")
            .with_f64_data(&data)
            .with_shape(&[rows, cols])
            .with_chunks(&[128, 128]);
    });
    // Shuffle + deflate: adds the shuffle filter and inflate to the assembly.
    let shuf_def = build(|b| {
        b.create_dataset("d")
            .with_f64_data(&data)
            .with_shape(&[rows, cols])
            .with_chunks(&[128, 128])
            .with_shuffle()
            .with_deflate(DeflateLevel::Six);
    });

    let plain_file = File::from_bytes(plain).unwrap();
    let shuf_def_file = File::from_bytes(shuf_def).unwrap();

    let mut g = c.benchmark_group("chunked_read_1024x1024_f64");
    g.bench_function("uncompressed", |b| {
        b.iter(|| std::hint::black_box(plain_file.dataset("d").unwrap().read_f64().unwrap()))
    });
    g.bench_function("shuffle_deflate", |b| {
        b.iter(|| std::hint::black_box(shuf_def_file.dataset("d").unwrap().read_f64().unwrap()))
    });
    g.finish();
}

// ---------------------------------------------------------------------------
// Write path
// ---------------------------------------------------------------------------

fn bench_write(c: &mut Criterion) {
    let n = 1 << 20;
    let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.25).collect();
    let rows = 1024u64;
    let cols = 1024u64;
    let data2d: Vec<f64> = (0..(rows * cols)).map(|i| i as f64).collect();

    let mut g = c.benchmark_group("write_1M_f64");
    g.bench_function("contiguous", |b| {
        b.iter(|| {
            std::hint::black_box(build(|bld| {
                bld.create_dataset("d")
                    .with_f64_data(std::hint::black_box(&data));
            }))
        })
    });
    g.bench_function("chunked_shuffle_deflate", |b| {
        b.iter(|| {
            std::hint::black_box(build(|bld| {
                bld.create_dataset("d")
                    .with_f64_data(std::hint::black_box(&data2d))
                    .with_shape(&[rows, cols])
                    .with_chunks(&[128, 128])
                    .with_shuffle()
                    .with_deflate(DeflateLevel::Six);
            }))
        })
    });
    g.finish();
}

// ---------------------------------------------------------------------------
// MATLAB v7.3 serde round-trip (write transpose + numeric seq decode)
// ---------------------------------------------------------------------------

/// Exercise the MAT serde path end to end: serializing a struct holding a large
/// 2-D matrix (column-major transpose) and large numeric vectors, then reading
/// it back (per-element sequence decode). This is the path `mat::to_bytes` /
/// `mat::from_bytes` users actually pay for; the low-level benches above bypass
/// it. No-op unless the `serde` feature is enabled.
#[cfg(feature = "serde")]
fn bench_mat_roundtrip(c: &mut Criterion) {
    use hdf5_pure::mat::{self, Matrix};
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    struct Payload {
        matrix: Matrix<f64>,
        samples: Vec<f64>,
        labels: Vec<i32>,
    }

    let rows = 512usize;
    let cols = 512usize;
    let matrix = Matrix::from_row_major(
        rows,
        cols,
        (0..rows * cols).map(|i| i as f64 * 0.5).collect(),
    );
    let samples: Vec<f64> = (0..(1usize << 20)).map(|i| i as f64).collect();
    let labels: Vec<i32> = (0..(1usize << 20)).map(|i| i as i32).collect();
    let payload = Payload {
        matrix,
        samples,
        labels,
    };
    let bytes = mat::to_bytes(&payload).expect("serialize payload");

    let mut g = c.benchmark_group("mat_roundtrip");
    g.bench_function("to_bytes", |b| {
        b.iter(|| std::hint::black_box(mat::to_bytes(std::hint::black_box(&payload)).unwrap()))
    });
    g.bench_function("from_bytes", |b| {
        b.iter(|| {
            let p: Payload = mat::from_bytes(std::hint::black_box(&bytes)).unwrap();
            std::hint::black_box(p)
        })
    });
    g.finish();
}

#[cfg(not(feature = "serde"))]
fn bench_mat_roundtrip(_c: &mut Criterion) {}

// ---------------------------------------------------------------------------
// The write-back page buffer (issue #308)
// ---------------------------------------------------------------------------
//
// Everything else recorded for that feature is a write *count*, and a count is
// not the axis it is sold on. Two doubts are worth measuring, and they pull in
// opposite directions:
//
//   - The buffer requires `SyncPolicy::OnClose`, which issues no `fsync` between
//     the repeat dirtyings of a page — so the kernel page cache already
//     coalesces them into one device write, and what the buffer removes is a few
//     hundred `pwrite` syscalls into an already-cached file.
//   - The crash mark it must be paired with costs two `fsync`s per *session*,
//     one at open and one at close, whatever the session then does.
//
// So the buffer has a break-even session length, which is what these four
// benchmarks bracket rather than average away: `short` sits below it and `long`
// above it, and the pair is the measurement — a single length would report
// whichever answer it happened to sit on.
//
// Measured on an Apple M1 Max (APFS), 256-byte appends into eight datasets. A
// controlled sweep of the same workload — arms alternated, median of seven —
// puts the crossing near 800 appends:
//
//   |  appends | off      | on       | ratio |
//   |---------:|---------:|---------:|------:|
//   |      400 |  14.5 ms |  19.3 ms |  0.75 |
//   |      800 |  23.3 ms |  24.3 ms |  0.96 |
//   |    1,600 |  41.0 ms |  33.3 ms |  1.23 |
//   |    3,200 |  79.1 ms |  54.8 ms |  1.45 |
//   |    6,400 | 152.9 ms |  93.5 ms |  1.64 |
//
// Below the crossing the two `fsync`s cost more than the saved syscalls; above
// it the ratio climbs, because the same pages are re-dirtied more often the
// longer the session runs. The 1,600-append session ran 1.64x with the crash
// mark removed against 1.23x with it — that gap is what the guarantee costs, a
// fixed price per session rather than a rate. The margin also narrows to about
// 1.1x once 64 KiB payloads rather than metadata churn dominate.
//
// **Read the ratio between a pair, not the absolute times, and re-measure rather
// than trusting these constants.** The `off` arm issues thousands of small
// writes and is sensitive to background I/O: across runs on an otherwise-busy
// machine `short` ranged 0.65-1.11x and `long` 1.26-2.28x. The short arm is the
// noisier of the two, because both of its arms are small enough for the fixed
// `fsync` cost to dominate.

fn bench_page_buffer(c: &mut Criterion) {
    use hdf5_pure::{FileAccessProperties, FileLocking, FileSpaceStrategy, SyncPolicy};

    const DATASETS: usize = 8;
    const CHUNK: usize = 64;

    let dir = tempfile::tempdir().expect("temp dir");
    let names: Vec<String> = (0..DATASETS).map(|t| format!("t{t}")).collect();
    let batch = vec![0i32; CHUNK];

    let fixture = |path: &std::path::Path| {
        let mut b = FileBuilder::new();
        b.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_page_size(FileSpacePageSize::DEFAULT),
        );
        for name in &names {
            b.create_dataset(name)
                .with_i32_data(&vec![0i32; CHUNK])
                .with_shape(&[CHUNK as u64])
                .with_maxshape(&[MaxExtent::Unlimited])
                .with_chunks(&[CHUNK as u64]);
        }
        b.write(path).expect("fixture");
    };

    // Built once and copied per iteration, then forced to disk: rebuilding it
    // each time leaves the setup's own writes dirty when the timed region
    // starts, and the arm under test then pays to flush them — which lands
    // entirely on the arm that issues more `fsync`s, and is exactly the arm
    // being judged.
    let template = dir.path().join("template.h5");
    fixture(&template);

    let mut g = c.benchmark_group("page_buffer");
    // Locking is disabled so the measurement is of the buffer rather than of
    // `flock`; every arm pays the same either way.
    for (length, rounds) in [("short", 50usize), ("long", 400)] {
        for (label, budget) in [("off", 0usize), ("on", 4 << 20)] {
            g.bench_function(format!("{length}/{label}"), |b| {
                b.iter_batched(
                    || {
                        let path = dir.path().join(format!("{length}_{label}.h5"));
                        let _ = std::fs::remove_file(&path);
                        std::fs::copy(&template, &path).expect("fixture copy");
                        std::fs::File::open(&path)
                            .and_then(|f| f.sync_all())
                            .expect("settle the fixture");
                        path
                    },
                    |path| {
                        let props = FileAccessProperties::new()
                            .with_sync_policy(SyncPolicy::OnClose)
                            .with_locking(FileLocking::Disabled)
                            .with_page_buffer_size(budget);
                        let file = File::open_rw_with_options(&path, props).unwrap();
                        for _ in 0..rounds {
                            for name in &names {
                                let mut ds = file.dataset(name).unwrap();
                                ds.append(std::hint::black_box(&batch)).unwrap();
                            }
                        }
                        file.close().unwrap();
                    },
                    criterion::BatchSize::PerIteration,
                );
            });
        }
    }
    g.finish();
}

fn bench_reusable_space_stats(c: &mut Criterion) {
    let directory = tempfile::tempdir().unwrap();
    let mut group = c.benchmark_group("reusable_space_stats");
    for strategy in [FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page] {
        for holes in [16, 128, 1_024] {
            let path = directory.path().join(format!("{strategy:?}-{holes}.h5"));
            let file = file_space::fragmented_session(&path, strategy, holes);
            let sections = file.reusable_free_sections().unwrap().section_count();
            assert!(sections >= holes / 2);
            group.bench_function(
                BenchmarkId::new(format!("{strategy:?}"), sections),
                |bench| {
                    bench.iter(|| std::hint::black_box(file.reusable_space_stats().unwrap()));
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_contiguous_decode,
    bench_primitive_decode,
    bench_primitive_encode,
    bench_primitive_append,
    bench_chunked_assembly,
    bench_write,
    bench_mat_roundtrip,
    bench_page_buffer,
    bench_reusable_space_stats
);
criterion_main!(benches);
