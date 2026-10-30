//! Allocation and timing measurements for metadata-heavy repacks.
//!
//! The workload consists of repeated subtrees with mostly empty chunked datasets, including
//! compressed and uncompressed datasets. Fixture creation and output verification run outside the
//! measured operation. Verification includes values, metadata, and encoded chunk bytes.
//!
//! Memory mode records allocations during repacking and writes HTML and DHAT reports in the target
//! directory. Timing mode measures warmed repacks with allocation recording disabled.
//!
//! Run either mode in an optimized build:
//!
//! ```text
//! just soundness::repack-profile memory
//! just soundness::repack-profile timing
//! ```
//!
//! The optional arguments after the mode select the number of subtrees and timing rounds.

use std::error::Error as StdError;
use std::fs::File as FsFile;
use std::hint;
use std::io::{Read, Seek, SeekFrom};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hdf5_pure::{AttrValue, Chunk, DeflateLevel, File, FileBuilder, MaxExtent, RepackOptions};

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

fn main() -> Result<(), Box<dyn StdError>> {
    let mut args = std::env::args().skip(1);
    let mode = match args.next().as_deref() {
        None | Some("memory") => Mode::Memory,
        Some("timing") => Mode::Timing,
        Some(value) => return Err(format!("unknown mode {value:?}; use memory or timing").into()),
    };
    let subtrees: NonZeroUsize = args.next().unwrap_or_else(|| "512".into()).parse()?;
    let rounds: NonZeroUsize = args.next().unwrap_or_else(|| "5".into()).parse()?;
    if args.next().is_some() {
        return Err("usage: repack_profile [memory|timing] [subtrees] [rounds]".into());
    }

    let dir = tempfile::tempdir()?;
    let src = dir.path().join("source.h5");
    let dst = dir.path().join("destination.h5");
    write_source(&src, subtrees.get())?;
    println!(
        "subtrees={}, groups={}, datasets={}, source_bytes={}",
        subtrees.get(),
        subtrees.get() * (GROUPS + 1) + 1,
        subtrees.get() * DATASETS,
        std::fs::metadata(&src)?.len(),
    );

    match mode {
        Mode::Memory => profile_memory(&src, &dst)?,
        Mode::Timing => {
            hdf5_pure::repack(&src, &dst, &RepackOptions::new())?;
            for round in 0..rounds.get() {
                let start = Instant::now();
                hdf5_pure::repack(&src, &dst, &RepackOptions::new())?;
                println!(
                    "round={}, elapsed_us={}",
                    round + 1,
                    start.elapsed().as_micros()
                );
            }
        }
    }
    verify(&src, &dst, subtrees.get())?;
    Ok(())
}

enum Mode {
    Memory,
    Timing,
}

fn write_source(path: &Path, subtrees: usize) -> Result<(), Box<dyn StdError>> {
    let mut file = FileBuilder::new();
    for subtree in 0..subtrees {
        let mut root = file.create_group(&format!("tree{subtree:05}"));
        for group in 0..GROUPS {
            let mut child = root.create_group(&format!("group{group:02}"));
            for slot in 0..(1 + usize::from(group < DATASETS - GROUPS)) {
                let index = group + slot * GROUPS;
                let values = values(index);
                let dataset = child.create_dataset(&format!("data{index:02}"));
                dataset
                    .with_i32_data(&values)
                    .with_shape(&[u64::try_from(values.len())?])
                    .with_maxshape(&[MaxExtent::Unlimited])
                    .with_chunks(&[CHUNK_ELEMENTS])
                    .set_attr("index", AttrValue::I32(i32::try_from(index)?));
                if index.is_multiple_of(3) {
                    dataset.with_deflate(DeflateLevel::Six);
                }
            }
            root.add_group(child.finish());
        }
        file.add_group(root.finish());
    }
    file.write(path)?;
    Ok(())
}

fn values(index: usize) -> Vec<i32> {
    if index.is_multiple_of(8) {
        (0..ELEMENTS).collect()
    } else {
        Vec::new()
    }
}

#[expect(
    clippy::panic_in_result_fn,
    reason = "the profile checks measurement validity"
)]
fn profile_memory(src: &Path, dst: &Path) -> Result<(), Box<dyn StdError>> {
    let profiler = heapscope::Profiler::builder().no_output().build()?;
    let control = {
        let _region = heapscope::region("allocation control");
        hint::black_box(vec![0_u8; CONTROL_BYTES])
    };
    let snapshot = heapscope::Snapshot::capture();
    check_snapshot(&snapshot);
    let control_stats = snapshot
        .regions
        .iter()
        .find(|region| region.name.as_deref() == Some("allocation control"))
        .expect("the entered control region is recorded");
    assert_eq!(control_stats.counts.curr_bytes, CONTROL_BYTES as u64);
    assert_eq!(control_stats.counts.max_bytes, CONTROL_BYTES as u64);
    drop(control);
    drop(snapshot);

    {
        let _region = heapscope::region("repack metadata");
        hdf5_pure::repack(src, dst, &RepackOptions::new())?;
    }
    let snapshot = heapscope::Snapshot::capture();
    check_snapshot(&snapshot);
    let stats = snapshot
        .regions
        .iter()
        .find(|region| region.name.as_deref() == Some("repack metadata"))
        .expect("the entered repack region is recorded");
    assert!(stats.counts.total_blocks > 0);
    assert!(stats.counts.max_bytes >= CHUNK_ELEMENTS * 4);
    println!(
        "repack: allocations={}, allocated_bytes={}, peak_bytes={}, live_bytes={}",
        stats.counts.total_blocks,
        stats.counts.total_bytes,
        stats.counts.max_bytes,
        stats.counts.curr_bytes,
    );
    let path = profile_path()?;
    snapshot.save_html(&path)?;
    snapshot.save_dhat_v2(path.with_extension("dhat.json"))?;
    snapshot.write_text_summary(std::io::stderr().lock(), 24)?;
    println!("profile={}", path.display());
    drop(profiler);
    Ok(())
}

fn check_snapshot(snapshot: &heapscope::Snapshot) {
    assert!(snapshot.exact, "the allocation snapshot is inconsistent");
    assert!(!snapshot.poisoned, "the allocation profiler failed");
    assert_eq!(snapshot.stats.dropped_blocks, 0);
    assert_eq!(snapshot.rows_dropped, 0);
    assert_eq!(snapshot.points_dropped, 0);
    assert_eq!(snapshot.unattributed_blocks, 0);
    assert!(
        snapshot
            .points
            .iter()
            .all(|point| point.kind == heapscope::output::PointKind::Recorded),
        "the allocation call-site table overflowed"
    );
}

#[expect(
    clippy::disallowed_methods,
    reason = "the executable path locates profile output without subprocess dispatch"
)]
fn profile_path() -> Result<PathBuf, Box<dyn StdError>> {
    let exe = std::env::current_exe()?;
    let target = exe.ancestors().nth(3).ok_or("missing target directory")?;
    Ok(target.join("repack-profile.html"))
}

#[expect(
    clippy::panic_in_result_fn,
    reason = "the profile checks repacked values and metadata"
)]
fn verify(src: &Path, dst: &Path, subtrees: usize) -> Result<(), Box<dyn StdError>> {
    let source = File::open_streaming(src)?;
    let output = File::open_streaming(dst)?;
    let mut source_bytes = FsFile::open(src)?;
    let mut output_bytes = FsFile::open(dst)?;
    assert_eq!(output.root().groups()?, source.root().groups()?);
    assert_eq!(output.root().groups()?.len(), subtrees);
    for subtree in 0..subtrees {
        let root = format!("tree{subtree:05}");
        assert_eq!(output.group(&root)?.groups()?.len(), GROUPS);
        for index in 0..DATASETS {
            let path = format!("{root}/group{:02}/data{index:02}", index % GROUPS);
            let before = source.dataset(&path)?;
            let after = output.dataset(&path)?;
            assert_eq!(after.read_i32()?, values(index));
            assert_eq!(after.shape()?, before.shape()?);
            assert_eq!(after.maxshape()?, before.maxshape()?);
            assert_eq!(after.chunk_shape()?, Some(vec![CHUNK_ELEMENTS]));
            assert_eq!(after.filter_pipeline()?, before.filter_pipeline()?);
            assert_eq!(after.attrs()?, before.attrs()?);
            let before_chunks = before.chunks()?;
            let after_chunks = after.chunks()?;
            assert_eq!(after_chunks.len(), before_chunks.len());
            for (before, after) in before_chunks.iter().zip(&after_chunks) {
                assert_eq!(after.offset, before.offset);
                assert_eq!(after.filter_mask(), before.filter_mask());
                assert_eq!(
                    chunk_bytes(&mut output_bytes, after)?,
                    chunk_bytes(&mut source_bytes, before)?
                );
            }
        }
    }
    Ok(())
}

fn chunk_bytes(file: &mut FsFile, chunk: &Chunk) -> Result<Vec<u8>, Box<dyn StdError>> {
    file.seek(SeekFrom::Start(chunk.address))?;
    let mut bytes = vec![0; usize::try_from(chunk.storage_size)?];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

const GROUPS: usize = 28;
const DATASETS: usize = 32;
const ELEMENTS: i32 = 512;
const CHUNK_ELEMENTS: u64 = 128;
const CONTROL_BYTES: usize = 4096;
