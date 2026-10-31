//! Replays completed file-write prefixes and checks allocation publication ordering.
//!
//! Prefix sweeps reconstruct each file from its starting bytes and recorded writes and resizes.
//! Each prefix must remain readable with an allowed data prefix, and replaying the complete log
//! must reproduce the completed workload. The sweeps cover EA appends, deferred initialization,
//! filtered tail replacement, and staged commits.
//!
//! The ordering tests also inspect synchronization outcomes at the storage boundary. For completed
//! full-chunk workloads, the tests locate content, ownership publication, and reference writes by
//! parsing chunk and index ranges, layout pointers, managers, and EOF generations. The tests use the
//! same discovery for altered traces whose files remain readable. These tests check requested
//! barriers and reported completion, without modeling physical power loss, torn metadata writes,
//! or SWMR visibility.

use std::path::Path;

use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::ExtensibleArrayHeaderLocator;
use test_util::disk_log::{self, DiskOp};
use test_util::object_header::v2;

use crate::FileSpaceSettings;
use crate::PersistedFreeSpacePolicy;
use crate::access_mode::AccessMode;
use crate::address::{BaseAddress, BaseAddressExt};
use crate::chunk_index_inplace::alloc_probe;
use crate::data_layout::{ChunkIndexLayout, DataLayout};
use crate::dataspace::MaxExtent;
use crate::edit::{AppendBuilder, AppendTarget, MemoryStrategy, SyncPolicy, WriteEngine};
use crate::error::{Error, FormatError};
use crate::file_lock::FileLocking;
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::FileSpaceStrategy;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;
use crate::type_builders::DatasetBuilder;
use crate::width::FormatWidths;
use crate::writer::FileBuilder;

mod ordering;
mod ordering_facts;

/// What reading one replayed prefix made of it.
enum Verdict {
    /// Read back, and the state is one a crash at this point may leave.
    Clean,
    /// Refused to read. Benign in general — the caller is told rather than
    /// misled — but see [`Tally::assert_sound`]: the operations swept here
    /// promise that the previous value stays readable, so this is a failure too.
    Loud(String),
    /// Read back without complaint, and returned something else.
    Silent(String),
}

impl Verdict {
    /// Classify a read. An `Error` is [`Loud`](Verdict::Loud) whatever it says —
    /// the file declined to hand over data, which is the outcome that cannot
    /// mislead. Only a *successful* read is judged on its contents.
    fn of<T>(read: Result<T, Error>, judge: impl FnOnce(T) -> Result<(), String>) -> Verdict {
        match read {
            Err(e) => Verdict::Loud(std::format!("{e:?}")),
            Ok(v) => match judge(v) {
                Ok(()) => Verdict::Clean,
                Err(why) => Verdict::Silent(why),
            },
        }
    }
}

/// A session's disk operations, and the file they started from.
struct Recording {
    label: String,
    /// The file's bytes before the recorded session opened it. Read before the
    /// session exists rather than alongside it, so no platform's file locking
    /// has an opinion about it.
    base: Vec<u8>,
    /// The file the recorded session actually wrote, kept so the last prefix can
    /// be compared against it.
    real: std::path::PathBuf,
    ops: Vec<DiskOp>,
    /// Fresh Extensible-Array blocks the recorded window allocated, by kind.
    /// See [`assert_allocated`](Self::assert_allocated).
    data_blocks: usize,
    super_blocks: usize,
}

impl Recording {
    /// Record everything `work` puts on the disk. `work` is handed `path`, and
    /// must open and close its own session inside the call: a session's teardown
    /// writes are part of what a crash can interrupt, so they belong in the log.
    fn of(label: &str, path: &Path, work: impl FnOnce(&Path)) -> Self {
        let base = std::fs::read(path).expect("the starting file");
        // Discard whatever the warm-up allocated, so the counts below describe
        // the recorded window rather than the whole test.
        alloc_probe::take();
        disk_log::start();
        work(path);
        let ops = disk_log::take();
        let (data_blocks, super_blocks) = alloc_probe::take();
        // The recorded order itself, on request. This is what shows the
        // address-order inversion directly: with a barrier the block at
        // end-of-file is issued before the pointer naming it, and without one the
        // same two writes come out the other way round.
        if std::env::var_os("CRASH_REPLAY_OPS").is_some() {
            for (i, op) in ops.iter().enumerate() {
                std::eprintln!("OP {i} {}", op.describe());
            }
        }
        assert!(
            !ops.is_empty(),
            "{label}: recorded nothing, so there is no crash point to replay"
        );
        Self {
            label: label.to_string(),
            base,
            real: path.to_path_buf(),
            ops,
            data_blocks,
            super_blocks,
        }
    }

    /// Require that this fixture is positioned where the defect is visible. Two
    /// separate conditions, because the first draft of this module satisfied
    /// neither while sweeping five hundred prefixes and reporting every one
    /// clean:
    ///
    /// 1. The window **allocated** fresh index blocks of each kind, since the two
    ///    ordering barriers in [`crate::chunk_index_inplace`] are only reached by
    ///    an allocation. A window over the wrong rounds crosses neither.
    /// 2. End-of-file is **far enough from the index** that the two sides of a
    ///    publish cannot share a gathered write. This is the condition that is
    ///    easy to lose by retuning a constant: [`CHUNK`] used to be one element,
    ///    which kept the whole file inside two pages, and two writes that merge
    ///    into one are atomic — the inversion still happens and no prefix can
    ///    observe it. Deleting a barrier and shrinking `CHUNK` back to 1 makes
    ///    the sweeps pass again with condition 1 still satisfied, so nothing but
    ///    this catches it.
    fn assert_positioned(&self, data_blocks: usize, super_blocks: usize) {
        assert!(
            self.data_blocks >= data_blocks && self.super_blocks >= super_blocks,
            "{}: the recorded window allocated {} data block(s) and {} super block(s), \
             short of the {data_blocks} and {super_blocks} it is positioned for. \
             Nothing below can see a barrier that is not crossed.",
            self.label,
            self.data_blocks,
            self.super_blocks
        );
        let pages = std::fs::metadata(&self.real).unwrap().len() / GATHER_PAGE;
        assert!(
            pages >= MIN_PAGES,
            "{}: the file spans {pages} gather pages of {GATHER_PAGE} bytes, under the \
             {MIN_PAGES} this needs. Below that, a publish point near the front of the \
             file and the block it names at end-of-file merge into one gathered write, \
             which is atomic, and no replayed prefix can fall between them.",
            self.label
        );
    }

    /// Demand that a *publish* write in this recording crosses a gather-page
    /// boundary — that the structure being published really is wider than the
    /// page whose merging hid issue #307.
    ///
    /// A publish is a patch near the front of the file, so this looks for a write
    /// that begins inside the first gather page and reaches past it. Any write is
    /// not enough: an append at end-of-file crosses page boundaries all day and
    /// says nothing about the header's width.
    ///
    /// Without this, a padded fixture is one object-header layout change away
    /// from sitting wholly inside a page, where every publish is atomic for free
    /// and the sweep passes while proving nothing. The padding was measured, and
    /// a measured constant is exactly the kind that drifts.
    fn assert_publishes_across_a_gather_page(&self) {
        let crossing = self.ops.iter().any(|op| match *op {
            DiskOp::Write { offset, ref bytes } => {
                offset < GATHER_PAGE && offset + bytes.len() as u64 > GATHER_PAGE
            }
            DiskOp::SetLen(_) => false,
        });
        assert!(
            crossing,
            "{}: no write starts inside the first {GATHER_PAGE}-byte gather page and \
             reaches past it, so the header this sweep publishes into fits in one page \
             and every publish is atomic whatever the engine does. The fixture needs a \
             wider object header.",
            self.label
        );
    }

    /// Materialize every prefix in turn and hand each to `check`, then hold it to
    /// `expect`.
    ///
    /// `finished` is what the *last* prefix must satisfy — the state the workload
    /// was supposed to reach. Without it every sweep here is satisfiable by doing
    /// nothing: prefix 0 is the untouched starting file, every checker accepts it
    /// (an append that has not happened and a commit that has not landed are both
    /// legitimate crash states), and so a regression that silently stopped
    /// publishing would leave all five sweeps reporting 100% clean.
    fn replay_every_prefix(
        &self,
        dir: &Path,
        check: impl Fn(&Path) -> Verdict,
        finished: impl Fn(&Path) -> Result<(), String>,
    ) -> Tally {
        let path = dir.join(std::format!("{}.replay.h5", self.label));
        let mut image = self.base.clone();
        let mut tally = Tally {
            total: self.ops.len() + 1,
            clean: 0,
            loud: Vec::new(),
            silent: Vec::new(),
        };

        for k in 0..=self.ops.len() {
            if k > 0 {
                self.ops[k - 1].apply(&mut image);
            }
            std::fs::write(&path, &image).unwrap();
            if k == self.ops.len() {
                // Replaying the whole log must reproduce the file the session
                // actually left. Anything reaching the disk outside
                // `BufferedWrites` — a raw positioned write on the handle, a
                // second thread — is invisible to the log, and every prefix
                // above would then be a state that never existed.
                assert_eq!(
                    image,
                    std::fs::read(&self.real).unwrap(),
                    "{}: replaying every recorded operation does not reproduce the \
                     file the session left, so the log is not everything that \
                     reached the disk",
                    self.label
                );
                if let Err(why) = finished(&path) {
                    panic!(
                        "{}: the completed workload did not reach the state this \
                         sweep is judging interruptions of: {why}",
                        self.label
                    );
                }
            }
            let after = if k == 0 {
                "the starting file".to_string()
            } else {
                self.ops[k - 1].describe()
            };
            match check(&path) {
                Verdict::Clean => tally.clean += 1,
                Verdict::Loud(why) => tally
                    .loud
                    .push(std::format!("after op {k} ({after}): {why}")),
                Verdict::Silent(why) => tally
                    .silent
                    .push(std::format!("after op {k} ({after}): {why}")),
            }
        }
        std::fs::remove_file(&path).ok();
        // How wide the sweep actually was, for whoever is changing it. The
        // assertions below are rules rather than counts — an exact figure here is
        // a property of one platform's allocator and page size — so the numbers
        // are printed on request instead of pinned.
        if std::env::var_os("CRASH_REPLAY_STATS").is_some() {
            std::eprintln!(
                "{}: {} prefixes, {} clean, {} loud, {} silent",
                self.label,
                tally.total,
                tally.clean,
                tally.loud.len(),
                tally.silent.len()
            );
        }
        tally.assert_sound(&self.label);
        tally
    }
}

/// What one sweep found, kept so a caller can compare two of them.
struct Tally {
    total: usize,
    clean: usize,
    /// Why each prefix refused to read. The benign outcome, but the reasons are
    /// what say *which* benign outcome, and they are the first thing to read when
    /// the floor below fails.
    loud: Vec<String>,
    silent: Vec<String>,
}

impl Tally {
    /// The two rules every sweep is held to.
    ///
    /// **Nothing silent.** A prefix that reads cleanly and hands back the wrong
    /// data is the outcome with no signal at all, and no crash point may produce
    /// one.
    ///
    /// **Nothing unreadable.** Stronger, and the real promise of an operation the
    /// crate calls crash-atomic: the dataset was readable before the operation
    /// began, so a crash part-way through must leave it readable, as the old
    /// value. A prefix that refuses to read is a window in which the file went
    /// from readable to not — a regression a caller experiences even though
    /// nothing is silently wrong.
    ///
    /// The second is what the ordering barriers buy, though not all of them
    /// through this rule. Removing the barrier before a fresh *data* block breaks
    /// it directly: the pointer is issued before the block, and the prefix between
    /// them reads `UnexpectedEof`, naming an address past end-of-file. The *super*
    /// block's barrier produces no unreadable prefix at all and is caught by the
    /// first rule instead, in [`a_crashed_append_can_be_reopened_and_appended_to`].
    fn assert_sound(&self, label: &str) {
        assert!(
            self.silent.is_empty(),
            "{label}: {} of {} replayed prefixes read cleanly and returned the wrong data:\n  {}",
            self.silent.len(),
            self.total,
            self.silent.join("\n  ")
        );
        assert!(
            self.loud.is_empty(),
            "{label}: {} of {} replayed prefixes refused to read, though every one \
             of them is a crash during an operation that leaves the previous value \
             in place and readable:\n  {}",
            self.loud.len(),
            self.total,
            self.loud.join("\n  ")
        );
    }
}

#[rstest::rstest]
#[case::immediate(InitializationWrite::Immediate)]
#[case::borrowed(InitializationWrite::Borrowed)]
#[case::owned(InitializationWrite::Owned)]
fn deferred_initialization_survives_every_completed_write(
    #[case] operation: InitializationWrite,
    #[values(false, true)] staged_creation: bool,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] paged: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] sync: SyncPolicy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("initialize.h5");
    let header = deferred_base_with_creation(&path, filtered, paged, staged_creation);
    let recording = Recording::of("initialize", &path, |path| {
        let file = crate::File::open_rw_with_options(
            path,
            crate::FileAccessProperties::new()
                .with_memory_strategy(backing)
                .with_sync_policy(sync),
        )
        .unwrap();
        operation.run(file.dataset("d").unwrap());
        file.close().unwrap();
    });
    let states = std::cell::Cell::new(0_u8);
    recording.replay_every_prefix(
        directory.path(),
        |path| {
            match inspect_deferred_pair(path, header) {
                Ok(allocated) => states.set(states.get() | if allocated { 2 } else { 1 }),
                Err(error) => return Verdict::Loud(std::format!("{error:?}")),
            }
            match appended_prefix_is_intact(path, 0, 11) {
                Verdict::Clean => {}
                other => return other,
            }
            match persisted_free_space_holds_nothing_live(path) {
                Verdict::Clean => {}
                other => return other,
            }
            let before = std::fs::read(path).unwrap();
            let file = crate::File::open(path).unwrap();
            let values = file.dataset("d").unwrap().read_i32().unwrap();
            file.close().unwrap();
            assert_eq!(std::fs::read(path).unwrap(), before);
            if values.is_empty() && states.get() & 2 != 0 {
                states.set(states.get() | 4);
            }
            let resumed = crate::File::open_rw(path).and_then(|file| {
                file.dataset("d")?
                    .append(&(i32::try_from(values.len()).unwrap()..13).collect::<Vec<_>>())?;
                file.close()
            });
            if let Err(error) = resumed {
                return Verdict::Loud(std::format!("resuming initialization: {error:?}"));
            }
            appended_prefix_is_intact(path, 13, 13)
        },
        |path| appended_all_the_way(path, 11),
    );
    assert_eq!(
        states.get(),
        7,
        "unallocated, allocated empty, and data states must occur"
    );
}

enum InitializationWrite {
    Borrowed,
    Immediate,
    Owned,
}

impl InitializationWrite {
    fn run(self, mut target: crate::Dataset) {
        let values: Vec<i32> = (0..11).collect();
        match self {
            Self::Borrowed => {
                let mut appender = target.buffered_appender().unwrap();
                appender.append(&values).unwrap();
                appender.finish().unwrap();
            }
            Self::Immediate => target.append(&values).unwrap(),
            Self::Owned => {
                let mut appender = target.into_buffered_appender().unwrap();
                appender.append(&values).unwrap();
                appender.finish().unwrap();
            }
        }
    }
}

fn deferred_base(path: &Path, filtered: bool, paged: bool) -> usize {
    deferred_base_with_creation(path, filtered, paged, false)
}

fn deferred_base_with_creation(
    path: &Path,
    filtered: bool,
    paged: bool,
    staged_creation: bool,
) -> usize {
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(if paged {
            FileSpaceStrategy::Page
        } else {
            FileSpaceStrategy::FsmAggr
        })
        .with_persistence(true),
    );
    let configure = |dataset: &mut DatasetBuilder| {
        dataset
            .with_unallocated_storage(crate::make_i32_type(), &[0])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4]);
        if filtered {
            dataset.with_fletcher32();
        }
    };
    if staged_creation {
        builder.write(path).unwrap();
        let file = crate::File::open_rw(path).unwrap();
        file.root().create_dataset("d", configure).unwrap();
        file.commit().unwrap();
        file.close().unwrap();
    } else {
        configure(builder.create_dataset("d"));
        builder.write(path).unwrap();
    }
    let bytes = std::fs::read(path).unwrap();
    let (header, layout) = test_util::bytes::signature_offsets(&bytes, v2::SIGNATURE)
        .into_iter()
        .find_map(|at| {
            ObjectHeader::parse(&bytes, AccessMode::ReadOnly, at, 8, 8)
                .unwrap()
                .messages
                .into_iter()
                .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
                .map(|message| (at, message.data))
        })
        .unwrap();
    assert!(matches!(
        DataLayout::parse(&layout, 8, 8).unwrap(),
        DataLayout::Chunked {
            index: ChunkIndexLayout::ExtensibleArray { address: None, .. },
            ..
        }
    ));
    assert!(!inspect_deferred_pair(path, header).unwrap());
    header
}

fn inspect_deferred_pair(path: &Path, header: usize) -> Result<bool, String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    let object = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, header, 8, 8)
        .map_err(|error| error.to_string())?;
    let layout = object
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    let DataLayout::Chunked {
        index: ChunkIndexLayout::ExtensibleArray { address, .. },
        ..
    } = DataLayout::parse(&layout.data, 8, 8).map_err(|error| error.to_string())?
    else {
        return Err("expected Extensible Array layout".into());
    };
    let Some(address) = address else {
        return Ok(false);
    };
    let locator = ExtensibleArrayHeaderLocator::new(
        FormatWidths::from_sizes(8, 8).map_err(|error| error.to_string())?,
        BaseAddress::new(0),
        address,
    );
    let header_range = locator.parse_range().map_err(|error| error.to_string())?;
    let array = locator
        .parse(
            FileRegion::new(
                header_range,
                &bytes[usize::try_from(header_range.start().get()).unwrap()
                    ..usize::try_from(header_range.end().get()).unwrap()],
            )
            .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let index = array
        .index_block()
        .expect("a published header has its complete index block");
    let range = index.range();
    if range.start() != array.range().end() {
        return Err(format!(
            "index {range:?} does not follow header {:?}",
            array.range()
        ));
    }
    index
        .parse(
            FileRegion::new(
                range,
                &bytes[usize::try_from(range.start().get()).unwrap()
                    ..usize::try_from(range.end().get()).unwrap()],
            )
            .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let file = crate::File::open(path).map_err(|error| error.to_string())?;
    if range.end().get() > file.superblock().eof_address {
        return Err(format!(
            "published pair ends at {} past EOF {}",
            range.end().get(),
            file.superblock().eof_address
        ));
    }
    for section in file
        .persisted_free_sections()
        .map_err(|error| error.to_string())?
        .unwrap()
        .free_sections()
        .sections()
    {
        if section.end_address().get() > header_range.start().get()
            && range.end().get() > section.address().get()
        {
            return Err(format!(
                "published pair {header_range:?}..{range:?} overlaps {section:?}"
            ));
        }
    }
    file.close().map_err(|error| error.to_string())?;
    Ok(true)
}

// ---------------------------------------------------------------------------
// The append workload
// ---------------------------------------------------------------------------

/// The page a locked session merges its writes within, and the least number of
/// them a fixture must span. See [`Recording::assert_positioned`].
const GATHER_PAGE: u64 = FileSpacePageSize::DEFAULT.get();
const MIN_PAGES: u64 = 8;

/// Elements per chunk, and per recorded round: one whole chunk per append.
///
/// The Extensible Array allocates blocks by *chunk count*, so the chunk width is
/// free, and it is not free in the one way that matters here. With a
/// one-element chunk the whole file is a few kilobytes, the index block and
/// end-of-file share a gathered write, and a missing barrier between them is
/// **invisible** — the two writes coalesce into one, which is atomic. Widening
/// the chunk moves end-of-file away from the index without changing which round
/// allocates what, which is what puts a prefix between the pointer and the block
/// it names.
const CHUNK: u64 = 64;
/// Elements each recorded round appends: exactly one chunk.
const ROUND: i32 = CHUNK as i32;

/// Rounds the base file already holds when recording starts, and rounds recorded
/// after it.
///
/// The window is positioned, not picked. With the C library's Extensible-Array
/// defaults a fresh *data* block is allocated at chunk 4, 20, 52, 84, 116, 180,
/// 244, 308, ..., and a fresh *super* block at 244 and then not again until 500.
/// Recording chunks 230..=320 therefore crosses two data-block allocations and
/// the super-block one between them, which are the crate's two barrier sites.
///
/// A window at 270 — the obvious round number — crosses no super-block
/// allocation at all, and a sweep positioned there passes with both barriers
/// deleted while still replaying five hundred prefixes and calling every one of
/// them clean. [`crate::chunk_index_inplace::alloc_probe`] is what turns that
/// into a failure rather than a silently narrower sweep.
const WARMUP_ROUNDS: i32 = 230;
/// Recorded rounds.
const ROUNDS: i32 = 90;
/// Elements the base file holds when recording starts.
const WARMUP: i32 = WARMUP_ROUNDS * ROUND;

/// Build a chunked, unlimited `i32` dataset holding `0..n`, and append to it
/// until it holds `WARMUP`. The appends are what matter: a file *written* with
/// 270 elements has one index shape, and a file *appended* to 270 has the shape
/// this harness is about, with the blocks allocated in the order an append
/// allocates them.
fn warmed_base(path: &Path, paged: bool) {
    warmed_base_padded(path, paged, 0, None);
}

/// [`warmed_base`], with `pad` bytes of attribute in the dataset's object header.
///
/// Padding is a positioning tool, not decoration. `patch_dimension` writes the
/// dataspace dimension near the front of chunk 0 and the chunk checksum at its
/// end; the two are joined into one write only when they share a gather page. A
/// header under [`GATHER_PAGE`] therefore publishes atomically for free and no
/// prefix can fall between them, which is why the unpadded sweeps below cannot
/// see issue #307. Above it they split, and the crash state between them is a
/// new value under its old checksum.
fn warmed_base_padded(path: &Path, paged: bool, pad: usize, strategy: Option<FileSpaceStrategy>) {
    let mut b = FileBuilder::new();
    if paged {
        b.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_page_size(FileSpacePageSize::DEFAULT),
        );
    } else if let Some(strategy) = strategy {
        b.with_file_space_settings(FileSpaceSettings::new(strategy));
    }
    {
        let d = b.create_dataset("d");
        d.with_i32_data(&[0i32])
            .with_shape(&[1])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[CHUNK]);
        if pad > 0 {
            d.set_attr(
                "pad",
                crate::type_builders::AttrValue::AsciiString("x".repeat(pad)),
            );
        }
    }
    b.write(path).unwrap();

    let mut s = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    s.set_sync_policy(SyncPolicy::OnClose);
    let mut n = 1i32;
    while n < WARMUP {
        let take = ROUND.min(WARMUP - n);
        append(&mut s, n, take);
        n += take;
    }
    drop(s);
}

/// Append `count` elements continuing the sequence at `from`.
fn append(s: &mut WriteEngine, from: i32, count: i32) {
    let mut b = AppendBuilder::new();
    b.append_i32(&(from..from + count).collect::<Vec<_>>());
    s.append_inplace_gathered(AppendTarget::Path(&ObjectPathBuf::parse("d")), &b, 4)
        .unwrap();
}

/// Read `d` and require that whatever length the file claims, every element up to
/// it is the one that belongs there, and that the length is between the one the
/// recording started at and the one it ended at.
///
/// Element *i* holds the value *i*, so a stale byte, a hole read as a fill value,
/// or a block resurrected from a previous shape all show up as a wrong value
/// rather than having to be looked for. A file that simply has fewer elements is
/// a crash before that append published, which is the whole point of the
/// barriers.
fn appended_prefix_is_intact(path: &Path, lo: i32, hi: i32) -> Verdict {
    if let Err(why) = recorded_eof_covers_the_file(path) {
        return Verdict::Silent(why);
    }
    let read = crate::reader::File::open(path).and_then(|f| f.dataset("d")?.read_i32());
    Verdict::of(read, |data| {
        let n = data.len() as i32;
        if !(lo..=hi).contains(&n) {
            return Err(std::format!(
                "length {n} is outside the {lo}..={hi} this crash point can produce"
            ));
        }
        if let Some(i) = (0..data.len()).find(|&i| data[i] != i as i32) {
            return Err(std::format!(
                "element {i} is {} rather than {i} (length {n})",
                data[i]
            ));
        }
        Ok(())
    })
}

/// The superblock's recorded end-of-file must not name bytes the file does not
/// have.
///
/// Reading one dataset cannot see this. The recorded end-of-file is the
/// allocator's cursor, not something a read follows, so a superblock claiming
/// more file than exists decodes perfectly and returns every correct value — and
/// then the next session allocates from a cursor past the real end, placing a
/// block in a hole.
///
/// It is the check the three barriers in the first half of
/// [`apply_ea_append`](crate::chunk_index_inplace::apply_ea_append) exist for,
/// and the reason they were invisible to the first version of this module: they
/// separate content at end-of-file from the *superblock field that covers it*,
/// which is at address 8-ish and therefore first in address order. Every one of
/// them is caught here and nowhere else in the crate.
fn recorded_eof_covers_the_file(path: &Path) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| std::format!("reading the file back: {e}"))?;
    // A prefix whose superblock does not parse is a *loud* state, not this
    // function's business; the read below will classify it.
    let Ok(crate::superblock::LocatedSuperblock { superblock: sb, .. }) =
        crate::superblock::read(bytes.as_slice())
    else {
        return Ok(());
    };
    if sb.eof_address > bytes.len() as u64 {
        return Err(std::format!(
            "the superblock records end-of-file at {} but the file is {} bytes: \
             every byte between is one a later session would allocate from and \
             never find",
            sb.eof_address,
            bytes.len()
        ));
    }
    Ok(())
}

/// The completed workload must have reached the length it was aiming at.
///
/// This is what the last replayed prefix is held to. Every *other* prefix
/// legitimately accepts a shorter dataset — that is what a crash mid-append
/// leaves — so without this the whole sweep is satisfied by a session that
/// appended nothing at all.
fn appended_all_the_way(path: &Path, hi: i32) -> Result<(), String> {
    let data = crate::reader::File::open(path)
        .and_then(|f| f.dataset("d")?.read_i32())
        .map_err(|e| std::format!("the finished file does not read: {e:?}"))?;
    if data.len() as i32 != hi {
        return Err(std::format!(
            "it holds {} elements rather than the {hi} the workload appended",
            data.len()
        ));
    }
    Ok(())
}

/// [`warmed_base`], with a **shuffle** filter on the dataset.
///
/// Shuffle rather than a compressor because this fixture has to stay wide: the
/// sweeps demand a file spanning [`MIN_PAGES`] gather pages, and the values here
/// are `0..n`, which deflate takes to a small fraction of a page. Shuffle stores
/// a chunk at its full size, so the file is as big as the unfiltered one — and
/// it makes the element a filtered record all the same (client id 1: address +
/// stored size + filter mask), which is what these sweeps are about.
///
/// It is also the most adversarial shape for the trailing-chunk rewrite of issue
/// #393. A re-encoded shuffle chunk is exactly as long as the one it replaces,
/// so an implementation that overwrote the old slot rather than allocating a
/// fresh one would always "fit" — and the prefix in the middle of that overwrite
/// is what the sweep below reads.
fn warmed_shuffled_base(path: &Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..CHUNK as i32).collect::<Vec<_>>())
        .with_shape(&[CHUNK])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[CHUNK])
        .with_shuffle();
    b.write(path).unwrap();

    let mut s = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    s.set_sync_policy(SyncPolicy::OnClose);
    let mut n = ROUND;
    while n < WARMUP {
        append(&mut s, n, ROUND);
        n += ROUND;
    }
    drop(s);
}

/// A *filtered* dataset grown by appends that leave a partial trailing chunk,
/// which the next append then re-encodes (issue #393).
///
/// Each round appends 30 elements and then 34: the first opens a partial chunk,
/// the second decodes it, extends it, re-encodes it into a fresh allocation and
/// repoints the index element that a reader at the old dimension is still
/// following. That repoint is the only write in this crate that changes where an
/// *already visible* chunk lives, so every prefix of it has to read back as the
/// old length with the old values — which is what the module's `Silent` rule
/// says, applied to a rewrite rather than an insert.
///
/// Positioned like the unfiltered sweeps, so the Extensible-Array block barriers
/// are crossed here too; a filtered array reaches them at the same element
/// indices, since the block geometry counts elements rather than bytes. Deleting
/// the barrier before a fresh data-block pointer reddens this sweep, which is
/// what says it is a sweep rather than decoration.
///
/// One thing it cannot see, and the reason the rewrite allocates rather than
/// overwriting: the re-encoded chunk is a single contiguous write, and this
/// module replays whole writes atomically, so an implementation that overwrote
/// the old slot when the new bytes happened to fit would pass here. The argument
/// against that one is the disk's rather than the log's — a torn sector inside
/// the old slot leaves a reader decoding a mix of two encodings, with no crash
/// point this harness can position.
#[test]
fn regrowing_a_filtered_trailing_chunk_survives_a_crash_at_every_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("append_filtered.h5");
    warmed_shuffled_base(&path);

    let rec = Recording::of("append-filtered-tail", &path, |p| {
        let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        for r in 0..ROUNDS {
            let base = WARMUP + r * ROUND;
            append(&mut s, base, 30);
            append(&mut s, base + 30, ROUND - 30);
        }
        drop(s);
    });

    rec.assert_positioned(2, 1);
    let hi = WARMUP + ROUNDS * ROUND;
    rec.replay_every_prefix(
        dir.path(),
        |p| appended_prefix_is_intact(p, WARMUP, hi),
        |p| appended_all_the_way(p, hi),
    );
}

/// The same sweep on a paged file, where the free-space managers are written and
/// re-homed at close and the allocator rounds to a page. That teardown is a
/// second set of low-address publish points, at the end of a session rather than
/// of an operation.
#[test]
fn appending_to_a_paged_file_survives_a_crash_at_every_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("append_paged.h5");
    warmed_base(&path, true);

    let rec = Recording::of("append-paged", &path, |p| {
        let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        for r in 0..ROUNDS {
            append(&mut s, WARMUP + r * ROUND, ROUND);
        }
        s.finalize_persist().unwrap();
        drop(s);
    });

    rec.assert_positioned(2, 1);
    let hi = WARMUP + ROUNDS * ROUND;
    rec.replay_every_prefix(
        dir.path(),
        |p| appended_prefix_is_intact(p, WARMUP, hi),
        |p| appended_all_the_way(p, hi),
    );
}

/// A file left by a crash must not merely *read* — it must be usable. Every
/// prefix that reads cleanly is reopened, appended to, and read again, which is
/// what catches a state that decodes but whose index no longer describes where
/// the next element goes.
///
/// This is the only sweep that catches the barrier before a fresh **super**
/// block. That one publishes a pointer into the index block while the block it
/// names is still in the buffer, and the dimension covering it has not been
/// written, so no read follows the pointer and every prefix looks perfect. The
/// next append follows it, reads an address out of bytes that were never
/// written, and tries to write eight bytes at 4397202087197605906.
#[test]
fn a_crashed_append_can_be_reopened_and_appended_to() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recover.h5");
    warmed_base(&path, false);

    const RECOVER_ROUNDS: i32 = ROUNDS;
    // Chunks, not elements. The recovery has to reach past the region the
    // crashed session was allocating rather than filling in behind it, and a
    // data block is `data_blk_min_elmts` = 16 *chunks*, so this crosses two of
    // them. Written as a count of `ROUND` because it was once a bare 40 back
    // when a chunk was one element, and widening `CHUNK` silently turned it into
    // two thirds of a single chunk — an append that advanced the dataset barely
    // at all while its comment still claimed to cross a block.
    const RECOVER_CHUNKS: i32 = 40;
    const RECOVER_APPEND: i32 = RECOVER_CHUNKS * ROUND;
    let rec = Recording::of("recover", &path, |p| {
        let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        for r in 0..RECOVER_ROUNDS {
            append(&mut s, WARMUP + r * ROUND, ROUND);
        }
        drop(s);
    });

    rec.assert_positioned(2, 1);
    let hi = WARMUP + RECOVER_ROUNDS * ROUND;
    rec.replay_every_prefix(
        dir.path(),
        |p| {
        // Read it first: a file that will not open is loud, and there is nothing
        // to recover.
        match appended_prefix_is_intact(p, WARMUP, hi) {
            Verdict::Clean => {}
            other => return other,
        }
        let before = crate::reader::File::open(p)
            .and_then(|f| f.dataset("d")?.read_i32())
            .expect("just read it")
            .len() as i32;
        // Caught rather than allowed to unwind, because one shape of this defect
        // panics rather than returning: a super-block pointer published without
        // the block gives the next append a garbage address read out of bytes
        // that were never written, and the image asserts on the write rather
        // than performing it. Letting that abort the sweep would report the
        // first occurrence and name no prefix; caught, every one is listed with
        // the write that produced it.
        let reopened = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled)?;
            s.set_sync_policy(SyncPolicy::OnClose);
            let mut b = AppendBuilder::new();
            b.append_i32(&(before..before + RECOVER_APPEND).collect::<Vec<_>>());
            s.append_inplace_gathered(AppendTarget::Path(&ObjectPathBuf::parse("d")), &b, 4)?;
            drop(s);
            Ok::<(), Error>(())
        }));
        // A file that read cleanly and then cannot be written to is *not* the
        // benign outcome, however loudly the append fails. The read said the file
        // was whole; the failure says an index pointer names a block that is not
        // there, which no read follows because the dimension covering it was
        // never published. That is the shape the ordering barriers exist to
        // prevent, and it is only ever visible from here.
        match reopened {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Verdict::Silent(std::format!(
                    "reads cleanly at {before} elements, but reopening and appending fails: {e:?}"
                ));
            }
            Err(panic) => {
                let why = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("(non-string panic)")
                    .to_string();
                return Verdict::Silent(std::format!(
                    "reads cleanly at {before} elements, but reopening and appending panics: {why}"
                ));
            }
        }
        appended_prefix_is_intact(p, before + RECOVER_APPEND, before + RECOVER_APPEND)
        },
        |p| appended_all_the_way(p, hi),
    );
}

/// Bytes of interior free space available across the recorded append rounds.
///
/// The retained ceiling prevents truncation of this hole.
const REUSE_HOLE: usize = 2 << 20;

/// Recorded rounds for that sweep. Fewer than [`ROUNDS`], because every prefix
/// writes out a file that carries the hole above.
const REUSE_ROUNDS: i32 = 20;

/// [`warmed_base`] on a file that persists its free space, with a hole of
/// [`REUSE_HOLE`] bytes already recorded in the on-disk managers when the
/// recorded window opens.
///
/// The delete that makes the hole runs in *this* session, which closes before the
/// recording starts. That is what lets the sweep reach the reuse branch without a
/// commit inside the window: a persisting session seeds its free list from the
/// managers at open, so the recorded session finds the hole rather than making
/// it.
fn warmed_base_with_a_persisted_hole(path: &Path) {
    let mut b = FileBuilder::new();
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    b.create_dataset("d")
        .with_i32_data(&[0i32])
        .with_shape(&[1])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[CHUNK]);
    b.create_dataset("victim")
        .with_i32_data(&vec![7i32; REUSE_HOLE / 4]);
    // Above the hole, so deleting the victim leaves an interior region rather
    // than a run reaching end-of-file that the commit would truncate away.
    b.create_dataset("ceiling").with_i32_data(&[9i32, 9, 9]);
    b.write(path).unwrap();

    let mut s = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    s.set_sync_policy(SyncPolicy::OnClose);
    let mut n = 1i32;
    while n < WARMUP {
        let take = ROUND.min(WARMUP - n);
        append(&mut s, n, take);
        n += take;
    }
    s.delete(&ObjectPathBuf::parse("victim")).unwrap();
    s.commit().unwrap();
    drop(s);
}

/// Checks that persisted free sections exclude every chunk of `d`.
///
/// Each replay prefix is inspected before any writer can reuse its free sections.
///
/// Returns [`Verdict::Loud`] if a manager does not parse, and [`Verdict::Silent`] if a free
/// region overlaps a chunk.
fn persisted_free_space_holds_nothing_live(path: &Path) -> Verdict {
    let Ok(f) = crate::reader::File::open(path) else {
        // A prefix whose file does not open at all is the read's business, not
        // this one's; it is classified there.
        return Verdict::Clean;
    };
    let Ok(chunks) = f.dataset("d").and_then(|d| d.chunks()) else {
        return Verdict::Clean;
    };
    Verdict::of(f.persisted_free_sections(), |report| {
        let Some(report) = report else {
            return Ok(());
        };
        for c in chunks {
            let (lo, hi) = (c.address, c.address + c.storage_size);
            for section in report.free_sections().sections() {
                let (addr, end) = (section.address().get(), section.end_address().get());
                if addr < hi && lo < end {
                    return Err(std::format!(
                        "the managers advertise [{addr}, {}) as free, which overlaps a live \
                         chunk of `d` at [{lo}, {hi})",
                        end
                    ));
                }
            }
        }
        Ok(())
    })
}

/// Checks the appended prefix, retained ceiling, and exclusion of live chunks from free sections.
fn inspect_reused_append_prefix(path: &Path, lo: i32, hi: i32) -> Verdict {
    match appended_prefix_is_intact(path, lo, hi) {
        Verdict::Clean => {}
        other => return other,
    }
    let ceiling = crate::reader::File::open(path).and_then(|f| f.dataset("ceiling")?.read_i32());
    match ceiling {
        Ok(v) if v == vec![9, 9, 9] => {}
        Ok(v) => return Verdict::Silent(std::format!("`ceiling` reads as {v:?}")),
        Err(e) => return Verdict::Loud(std::format!("{e:?}")),
    }
    persisted_free_space_holds_nothing_live(path)
}

#[test]
fn appending_into_a_persisted_hole_survives_a_crash_at_every_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reserved.h5");
    warmed_base_with_a_persisted_hole(&path);
    let before = std::fs::metadata(&path).unwrap().len();

    let rec = Recording::of("reserved", &path, |p| {
        let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        for r in 0..REUSE_ROUNDS {
            append(&mut s, WARMUP + r * ROUND, ROUND);
        }
        drop(s);
    });

    // File growth below the appended byte count establishes reuse of the interior hole.
    let after = std::fs::metadata(&path).unwrap().len();
    let appended = (REUSE_ROUNDS * ROUND) as u64 * 4;
    assert!(
        after < before + appended / 2,
        "the recorded session grew the file from {before} to {after} while appending \
         {appended} bytes, so it appended at end-of-file and no reservation was swept"
    );
    rec.assert_positioned(1, 0);

    let hi = WARMUP + REUSE_ROUNDS * ROUND;
    rec.replay_every_prefix(
        dir.path(),
        |p| inspect_reused_append_prefix(p, WARMUP, hi),
        |p| appended_all_the_way(p, hi),
    );
}

// ---------------------------------------------------------------------------
// The commit workload
// ---------------------------------------------------------------------------

/// Churn rounds in one recorded commit session.
const CHURN: i32 = 6;
/// Elements `d` starts the commit workload with, and gains per round.
///
/// Sized so the file spans well over [`MIN_PAGES`] gather pages, for the reason
/// [`Recording::assert_positioned`] gives: at the 64 and 16 this used to be, the
/// whole file was under 4 KB, every write merged with every other, and the sweep
/// could not observe an inversion it nonetheless produced.
const COMMIT_BASE: i32 = 4096;
const COMMIT_STEP: i32 = 1024;

/// Elements of `added{r}`, and of the `doomed{r}` it replaces. Different lengths
/// so a round that resurrected the wrong object reads as the wrong length rather
/// than as plausible data.
fn added_values(r: i32) -> Vec<i32> {
    (0..2048 + r).map(|i| i * 7 + r).collect()
}
fn doomed_values(r: i32) -> Vec<i32> {
    (0..1024 + r).map(|i| i * 3 + r).collect()
}

/// Elements of `slot` after `g` rounds have committed. `slot` is deleted and
/// recreated at its own path in one commit (issue #305), so unlike `added`/
/// `doomed` it is never absent — the generation is what says which commit a
/// prefix caught. Lengths differ per generation, so a prefix that resurrected
/// the previous object reads as the wrong length rather than as plausible data.
fn slot_values(g: i32) -> Vec<i32> {
    (0..512 + 37 * g).map(|i| i * 11 + g).collect()
}

/// Build the file the commit workload churns: one growable dataset, and one
/// `doomed{r}` per round for the delete half to consume.
fn commit_base(path: &Path, paged: bool) {
    let mut b = FileBuilder::new();
    if paged {
        b.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_page_size(FileSpacePageSize::DEFAULT),
        );
    }
    b.create_dataset("d")
        .with_i32_data(&(0..COMMIT_BASE).collect::<Vec<i32>>())
        .with_shape(&[COMMIT_BASE as u64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[COMMIT_STEP as u64]);
    for r in 0..CHURN {
        let v = doomed_values(r);
        b.create_dataset(&std::format!("doomed{r}"))
            .with_i32_data(&v)
            .with_shape(&[v.len() as u64]);
    }
    let v = slot_values(0);
    b.create_dataset("slot")
        .with_i32_data(&v)
        .with_shape(&[v.len() as u64]);
    b.write(path).unwrap();
}

/// Read a dataset that may legitimately not be in the file, distinguishing "no
/// such path" from "the path is there and will not decode".
///
/// The difference matters and is easy to lose: mapping every error to `None`
/// turns a commit that published a link to a half-written object header — a
/// genuinely torn commit — into "this round has not committed yet", which every
/// rule below accepts.
fn read_optional(f: &crate::reader::File, path: &str) -> Result<Option<Vec<i32>>, Error> {
    match f.dataset(path) {
        Ok(ds) => ds.read_i32().map(Some),
        Err(Error::Format(FormatError::PathNotFound(missing))) if missing == path => Ok(None),
        Err(e) => Err(e),
    }
}

/// `d`'s elements, then each round's `added{r}` and `doomed{r}` — `None` where
/// the path is not in the file at all — and finally `slot`, which every prefix
/// must have.
type CommitState = (
    Vec<i32>,
    Vec<Option<Vec<i32>>>,
    Vec<Option<Vec<i32>>>,
    Vec<i32>,
);

/// A commit publishes at its superblock write, so every object must read as
/// either its pre-commit or its post-commit state — never a mixture, and never
/// something that is neither.
///
/// Each round creates one object and deletes another, so a later round reuses
/// space an earlier one freed (issue #261). That is what makes "reads as
/// something that is neither" reachable rather than theoretical: a header
/// published over a region a previous commit released is a *readable* file whose
/// dataset returns the deleted object's bytes.
fn commit_state_is_one_or_the_other(path: &Path) -> Verdict {
    let read = (|| -> Result<CommitState, Error> {
        let f = crate::reader::File::open(path)?;
        let d = f.dataset("d")?.read_i32()?;
        let mut added = Vec::new();
        let mut doomed = Vec::new();
        for r in 0..CHURN {
            added.push(read_optional(&f, &std::format!("added{r}"))?);
            doomed.push(read_optional(&f, &std::format!("doomed{r}"))?);
        }
        // Not `read_optional`: a replacement leaves the path occupied at every
        // instant, so "no such path" is a failure here rather than a state, and
        // it has to reach the judge as one. `dataset` returning `PathNotFound`
        // is an `Error`, which `Verdict::of` would file as `Loud` — which
        // `Tally::assert_sound` also fails on, but under a reason about the file
        // refusing to read rather than about the path being gone.
        let slot = match f.dataset("slot") {
            Ok(ds) => ds.read_i32()?,
            Err(Error::Format(FormatError::PathNotFound { .. })) => Vec::new(),
            Err(e) => return Err(e),
        };
        Ok((d, added, doomed, slot))
    })();

    Verdict::of(read, |(d, added, doomed, slot)| {
        // `d` is judged on its own. An in-place append publishes at its own
        // phase 4, before the staged commit in the same round runs, so its length
        // says nothing about whether that commit landed — the two are separate
        // publish points and this must not conflate them.
        let n = d.len() as i32;
        if !(COMMIT_BASE..=COMMIT_BASE + CHURN * COMMIT_STEP).contains(&n) {
            return Err(std::format!(
                "`d` has {n} elements, outside the {COMMIT_BASE}..={} this session can produce",
                COMMIT_BASE + CHURN * COMMIT_STEP
            ));
        }
        if let Some(i) = (0..d.len()).find(|&i| d[i] != i as i32) {
            return Err(std::format!("`d`[{i}] is {} rather than {i}", d[i]));
        }

        for r in 0..CHURN {
            if let Some(a) = &added[r as usize] {
                if *a != added_values(r) {
                    return Err(std::format!(
                        "`added{r}` is present with the wrong bytes (len {}, wanted {})",
                        a.len(),
                        added_values(r).len()
                    ));
                }
            }
            if let Some(x) = &doomed[r as usize] {
                if *x != doomed_values(r) {
                    return Err(std::format!(
                        "`doomed{r}` is still present but its bytes have changed"
                    ));
                }
            }
            // The create and the delete are staged together and published by one
            // superblock write, so they land together or not at all. This is the
            // commit's atomicity, and the one thing a torn commit cannot satisfy.
            if added[r as usize].is_some() != doomed[r as usize].is_none() {
                return Err(std::format!(
                    "round {r} is half-committed: added{r} {}, doomed{r} {}",
                    if added[r as usize].is_some() {
                        "present"
                    } else {
                        "absent"
                    },
                    if doomed[r as usize].is_some() {
                        "present"
                    } else {
                        "absent"
                    }
                ));
            }
        }
        // Rounds commit in order, so the committed ones are a prefix. A later
        // round present while an earlier one is missing means a commit was
        // published over a state it did not build on.
        let committed: Vec<bool> = (0..CHURN).map(|r| added[r as usize].is_some()).collect();
        if let Some(r) = (1..CHURN as usize).find(|&r| committed[r] && !committed[r - 1]) {
            return Err(std::format!(
                "round {r} committed but round {} did not, though {r} came second",
                r - 1
            ));
        }

        // `slot` is deleted and recreated at its own path in the same commit, so
        // it is occupied at every instant: a prefix where the path is gone is a
        // window in which a replacement was two operations rather than one.
        if slot.is_empty() {
            return Err("`slot` is not in the file, though every commit that \
                        removes it puts its replacement back in the same commit"
                .to_string());
        }
        // And it is occupied by the generation this prefix's *other* evidence
        // says: rounds commit in order and each round replaces `slot` once, so
        // after `c` committed rounds `slot` holds generation `c`. This is the
        // rule that makes a replacement one linearization point rather than
        // merely an atomic-looking pair — a `slot` a generation ahead of or
        // behind the round count would mean the delete and the create were
        // published at different instants.
        let c = committed.iter().filter(|&&b| b).count() as i32;
        if slot != slot_values(c) {
            let found = (0..=CHURN).find(|&g| slot == slot_values(g));
            return Err(match found {
                Some(g) => {
                    std::format!("`slot` holds generation {g} while {c} round(s) have committed")
                }
                None => std::format!(
                    "`slot` holds {} elements, which is no generation this session writes \
                     ({c} round(s) have committed, so generation {c} was expected)",
                    slot.len()
                ),
            });
        }
        Ok(())
    })
}

/// Every round of the commit workload must actually have committed, or the sweep
/// above is judging interruptions of a session that did nothing. `d` must also
/// have grown by every round's append.
fn every_round_committed(path: &Path) -> Result<(), String> {
    let f = crate::reader::File::open(path)
        .map_err(|e| std::format!("the finished file does not open: {e:?}"))?;
    let d = f
        .dataset("d")
        .and_then(|d| d.read_i32())
        .map_err(|e| std::format!("the finished `d` does not read: {e:?}"))?;
    let want = COMMIT_BASE + CHURN * COMMIT_STEP;
    if d.iter().copied().ne(0..want) {
        return Err(std::format!("the finished `d` does not hold 0..{want}"));
    }
    for r in 0..CHURN {
        let added = read_optional(&f, &std::format!("added{r}"))
            .map_err(|error| std::format!("the finished `added{r}` does not read: {error:?}"))?;
        if added != Some(added_values(r)) {
            return Err(std::format!(
                "`added{r}` does not hold its committed values"
            ));
        }
        let doomed = read_optional(&f, &std::format!("doomed{r}"))
            .map_err(|error| std::format!("the finished `doomed{r}` does not read: {error:?}"))?;
        if doomed.is_some() {
            return Err(std::format!("`doomed{r}` was never deleted"));
        }
    }
    let slot = f
        .dataset("slot")
        .and_then(|d| d.read_i32())
        .map_err(|e| std::format!("the finished `slot` does not read: {e:?}"))?;
    if slot != slot_values(CHURN) {
        return Err(std::format!(
            "`slot` holds {} elements rather than generation {CHURN}'s {}",
            slot.len(),
            slot_values(CHURN).len()
        ));
    }
    Ok(())
}

/// One recorded churn round: grow a dataset in place, create another, delete a
/// third, and commit. The three edits take different routes to the disk — the
/// append patches the header in place, the create and the delete both go through
/// the staged commit — so one recording covers all of them.
fn churn(s: &mut WriteEngine, r: i32) {
    let from = COMMIT_BASE + r * COMMIT_STEP;
    let mut b = AppendBuilder::new();
    b.append_i32(&(from..from + COMMIT_STEP).collect::<Vec<_>>());
    s.append_inplace_gathered(AppendTarget::Path(&ObjectPathBuf::parse("d")), &b, 4)
        .unwrap();

    let v = added_values(r);
    let mut db = DatasetBuilder::new(&std::format!("added{r}"));
    db.with_i32_data(&v).with_shape(&[v.len() as u64]);
    s.stage_created_dataset(&ObjectPathBuf::parse(&std::format!("/added{r}")), db)
        .unwrap();
    s.delete(&ObjectPathBuf::parse(&std::format!("doomed{r}")))
        .unwrap();

    // Delete and recreate one path in the *same* commit (issue #305), beside the
    // unrelated create and delete above. This is the edit that has no two-commit
    // intermediate state to fall into, so a prefix where `slot` is missing is a
    // real failure rather than a legitimate half-done rotation.
    s.delete(&ObjectPathBuf::parse("slot")).unwrap();
    let v = slot_values(r + 1);
    let mut sb = DatasetBuilder::new("slot");
    sb.with_i32_data(&v).with_shape(&[v.len() as u64]);
    s.stage_created_dataset(&ObjectPathBuf::parse("/slot"), sb)
        .unwrap();

    s.commit().unwrap();

    // The workload has to have done all three, or the sweep is judging a session
    // that quietly did nothing.
    assert!(!s.has_staged_edits(), "the commit left edits staged");
}

/// The staged-commit path, swept over both backings and both file-space
/// strategies. A commit's publish point is the superblock at address zero, which
/// is the lowest address in the file and therefore the first thing gathering
/// wants to issue.
#[test]
fn committing_survives_a_crash_at_every_write() {
    for paged in [false, true] {
        for bounded in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("commit.h5");
            commit_base(&path, paged);
            let label = std::format!(
                "commit-{}-{}",
                if paged { "paged" } else { "plain" },
                if bounded { "bounded" } else { "mirrored" }
            );

            let rec = Recording::of(&label, &path, |p| {
                let mut s = if bounded {
                    WriteEngine::open_rw_with_strategy(
                        p,
                        MetadataCacheConfig::new(64 * 1024),
                        FileLocking::Enabled,
                        MemoryStrategy::Bounded,
                        PersistedFreeSpacePolicy::Tolerant,
                    )
                    .unwrap()
                } else {
                    WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap()
                };
                s.set_sync_policy(SyncPolicy::OnClose);
                for r in 0..CHURN {
                    churn(&mut s, r);
                }
                if paged {
                    s.finalize_persist().unwrap();
                }
                drop(s);
            });

            rec.assert_positioned(1, 0);
            rec.replay_every_prefix(
                dir.path(),
                commit_state_is_one_or_the_other,
                every_round_committed,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Gathering against not gathering
// ---------------------------------------------------------------------------

/// The guarantee does not depend on the write gathering.
///
/// It once did. A publish was a value write followed by a checksum write, joined
/// into one only where the gatherer found the two in the same page, so an
/// unbuffered session left this sweep 20 unreadable states in 168 — every one a
/// checksum torn from the value it covered. Since issue #307 the engine publishes
/// a checksummed structure as one write itself, and both configurations sweep
/// clean.
///
/// Running both is what keeps it that way. A publish point that regressed to two
/// writes would pass the gathered sweep for as long as the two shared a page, and
/// fail here at once — which is exactly how #307 hid: it was reachable only on a
/// dataset whose object header outgrew one page.
#[test]
fn crash_states_read_back_with_and_without_write_gathering() {
    use crate::image::WriteBuffering;

    const ROUNDS_EACH: i32 = 20;
    let hi = WARMUP + ROUNDS_EACH * ROUND;

    let sweep = |label: &str, mode: Option<WriteBuffering>| -> Tally {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("compare.h5");
        warmed_base(&path, false);
        let rec = Recording::of(label, &path, |p| {
            let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
            s.set_sync_policy(SyncPolicy::OnClose);
            if let Some(mode) = mode {
                s.set_write_buffering(mode).unwrap();
            }
            for r in 0..ROUNDS_EACH {
                append(&mut s, WARMUP + r * ROUND, ROUND);
            }
            drop(s);
        });
        rec.assert_positioned(1, 0);
        rec.replay_every_prefix(
            dir.path(),
            |p| appended_prefix_is_intact(p, WARMUP, hi),
            |p| appended_all_the_way(p, hi),
        )
    };

    // `None` keeps the session default, which is the gathering a locked
    // read-write session takes.
    let gathered = sweep("compare-gathered", None);
    let straight = sweep("compare-unbuffered", Some(WriteBuffering::Unbuffered));

    // Both sweeps assert the rule inside `replay_every_prefix`, so what is left
    // to pin here is that the second one is the *finer* of the two. Unbuffered
    // writing issues each patch separately, so it stops the machine at instants
    // the gathered sweep never reaches; if the two ever swept the same number of
    // prefixes, the mode would be being ignored and this whole comparison would
    // be one sweep run twice.
    assert!(
        straight.total > gathered.total,
        "the unbuffered sweep should stop at more instants than the gathered one, \
         but swept {} prefixes against {}",
        straight.total,
        gathered.total
    );
}

/// A super block is published one data-block pointer at a time, and the *second*
/// such publish lands in a super block a reader can already see — so a checksum
/// torn from the pointer it covers is a state a reader reaches, and since #312 a
/// state it refuses.
///
/// The first pointer into a *fresh* super block is not that state. The block
/// becomes reachable only once the header's element count is published, and by
/// then its checksum is whole — the ordering hides the window rather than the
/// reader missing it. So this sweep starts *past* the super block's own
/// allocation instead of across it, which is what the sweeps positioned at the
/// warm-up boundary do.
///
/// Unbuffered for the reason [`crash_states_read_back_with_and_without_write_gathering`]
/// gives: a super block is small, so gathering merges the two writes of a split
/// publish into one and hides exactly what this exists to catch.
#[test]
fn publishing_into_a_live_super_block_survives_a_crash_at_every_write() {
    use crate::image::WriteBuffering;

    /// The round to start recording at. With the C library's defaults a fresh
    /// data block is allocated at chunk 244 — which is also where the super
    /// block holding it is created — and the next at 308. Recording 300..=320
    /// therefore crosses a data-block allocation inside a super block that has
    /// held live elements since chunk 244.
    const START_ROUNDS: i32 = 300;
    const ROUNDS_EACH: i32 = 20;
    let lo = START_ROUNDS * ROUND;
    let hi = lo + ROUNDS_EACH * ROUND;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("live_super.h5");
    warmed_base(&path, false);
    {
        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        let mut n = WARMUP;
        while n < lo {
            append(&mut s, n, ROUND);
            n += ROUND;
        }
        drop(s);
    }

    let rec = Recording::of("live-super", &path, |p| {
        let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        s.set_write_buffering(WriteBuffering::Unbuffered).unwrap();
        for r in 0..ROUNDS_EACH {
            append(&mut s, lo + r * ROUND, ROUND);
        }
        drop(s);
    });
    // A floor of one data-block allocation. `assert_positioned` states only
    // floors, so the super-block *ceiling* is its own assertion: a window that
    // allocated one would be back at the case the ordering already hides, and
    // would sweep clean however the publish were written.
    rec.assert_positioned(1, 0);
    assert_eq!(
        rec.super_blocks, 0,
        "the window must publish *into* a live super block, not allocate one"
    );
    rec.replay_every_prefix(
        dir.path(),
        |p| appended_prefix_is_intact(p, lo, hi),
        |p| appended_all_the_way(p, hi),
    );
}

/// Bytes of attribute padding that push the dataspace dimension and its
/// object-header checksum into different gather pages. Measured threshold on the
/// default 4096-byte page: 3800 still joins, 3900 splits.
const HEADER_PAD: usize = 4096;

/// An in-place append publishes its new dimension and the checksum covering it.
/// A header chunk wider than one gather page splits those into two writes, and a
/// crash between them leaves the file **unreadable** — the new value under the
/// old checksum (issue #307).
#[test]
fn publishing_a_dimension_in_a_wide_header_survives_a_crash_at_every_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wide_header.h5");
    warmed_base_padded(&path, true, HEADER_PAD, None);

    let rec = Recording::of("wide-header", &path, |p| {
        let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        for r in 0..ROUNDS {
            append(&mut s, WARMUP + r * ROUND, ROUND);
        }
        s.finalize_persist().unwrap();
        drop(s);
    });

    rec.assert_publishes_across_a_gather_page();
    let hi = WARMUP + ROUNDS * ROUND;
    rec.replay_every_prefix(
        dir.path(),
        |p| appended_prefix_is_intact(p, WARMUP, hi),
        |p| appended_all_the_way(p, hi),
    );
}

/// Publishing costs the same number of writes whether or not the structure being
/// published spans a gather page — which is the defect of issue #307 stated as a
/// rule rather than as one of its consequences.
///
/// The sweep above catches the consequence, and only for the object header: it
/// reads the file back, so it sees a torn publish only where a reader follows the
/// torn field. This sees the split itself, so it covers every publish an append
/// makes — the array header's statistics, an index block's element and its data-
/// block pointers, a data block's element — without needing a workload that
/// reads each one.
///
/// Measured before the fix: 5 writes at 3800 bytes of padding against 6 at 3900,
/// the extra one being the checksum stranded in the next page. Both buffering
/// modes are checked, because the unbuffered one splits every publish regardless
/// of where the pages fall.
#[test]
fn a_publish_costs_the_same_whether_or_not_it_spans_a_page() {
    use crate::image::WriteBuffering;

    for mode in [None, Some(WriteBuffering::Unbuffered)] {
        let mut counts = Vec::new();
        // A sweep rather than the two sides of the measured threshold: picking
        // the boundary by hand is how a fixture ends up just short of the case it
        // was written for, and stepping across it costs milliseconds.
        for pad in [0usize, 2000, 3800, 3900, 4200, 8300] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("pad.h5");
            warmed_base_padded(&path, true, pad, None);

            let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
            s.set_sync_policy(SyncPolicy::OnClose);
            if let Some(mode) = mode {
                s.set_write_buffering(mode).unwrap();
            }
            let before = s.issued_write_order().len();
            append(&mut s, WARMUP, ROUND);
            let made = s.issued_write_order()[before..].to_vec();
            drop(s);
            // Whether this padding put the publish across a page: a write that
            // starts inside the first gather page and reaches past it.
            let wide = made
                .iter()
                .any(|&(off, len)| off < GATHER_PAGE && off + len > GATHER_PAGE);
            counts.push((pad, made.len(), wide));
        }
        // The comparison is only meaningful across the boundary. All-narrow or
        // all-wide passes with the defect present — measured — so the sweep has to
        // say it landed on both sides rather than be trusted to.
        assert!(
            counts.iter().any(|&(_, _, w)| w) && counts.iter().any(|&(_, _, w)| !w),
            "{mode:?}: every padding fell on the same side of the {GATHER_PAGE}-byte \
             page, so this comparison holds nothing: {counts:?}"
        );
        let (_, first, _) = counts[0];
        assert!(
            counts.iter().all(|&(_, n, _)| n == first),
            "{mode:?}: one append must cost the same writes at every header width, \
             but cost {counts:?}"
        );
    }
}

/// A publish writes back from the byte it changed, not from the front of the
/// structure that byte sits in.
///
/// Two appends into the same Extensible-Array data block touch a slot one
/// element further along each time, so the second writes strictly fewer bytes
/// than the first. Rewriting the whole block would make the two equal — which is
/// what this test exists to fail on, since the write *count* is the same either
/// way and every other check here counts writes.
///
/// Measured on this fixture: 531 bytes then 523, against 992 and 992 when the
/// publish starts at the structure instead. The block, not the object header, is
/// where the saving is: 534 bytes of index block against the 116 the first of
/// these appends actually changes.
#[test]
fn a_publish_writes_from_the_byte_it_changed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("from.h5");
    warmed_base(&path, false);

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    s.set_sync_policy(SyncPolicy::OnClose);
    let written = |s: &WriteEngine| -> u64 { s.issued_write_order().iter().map(|&(_, n)| n).sum() };
    let mut cost = Vec::new();
    for i in 0..2 {
        let before = written(&s);
        append(&mut s, WARMUP + i * ROUND, ROUND);
        cost.push(written(&s) - before);
    }
    drop(s);
    assert!(
        cost[1] < cost[0],
        "an append that touches a later slot of the same block must write fewer \
         bytes than one that touches an earlier slot, but the two cost {cost:?}"
    );
}
