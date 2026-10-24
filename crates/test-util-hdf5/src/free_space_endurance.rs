//! Exercises small-file reuse across commits and reopen checkpoints, with untracked growth as a
//! control.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fmt::{self, Debug, Display, Formatter};
use std::fs;
use std::path::Path;

use test_util::temp;

use super::api::{
    AttrValue, Error, File, FileAccessProperties, FileBuilder, FileSpaceInfo, FileSpacePageSize,
    FileSpaceStrategy, FormatError, Layout, MemoryStrategy, SyncPolicy,
};
use super::file_space;

/// Replaces groups repeatedly and reports file sizes and reusable free space.
///
/// Persistent strategies must stay within the size bound and reuse recovered holes. The
/// [`FileSpaceStrategy::None`] control must grow beyond that bound.
///
/// # Panics
///
/// Panics if a file operation fails or a workload assertion fails.
pub fn run_endurance(strategy: FileSpaceStrategy) -> String {
    run_endurance_inner(strategy, None::<fn(&Path, &str)>, |_, _| {})
}

/// Runs endurance with observations at each closed reopen checkpoint and after the final close.
///
/// The observer receives the closed file's path and a checkpoint label. It runs before the next
/// writing session opens. The returned report describes the final reopened state before its
/// editing session closes.
///
/// # Panics
///
/// Panics if a file operation fails, a workload assertion fails, or the observer panics.
pub fn run_endurance_observed(
    strategy: FileSpaceStrategy,
    observer: impl FnMut(&Path, &str),
) -> String {
    run_endurance_inner(strategy, Some(observer), |_, _| {})
}

/// Runs endurance with access to committed and reopened editing sessions.
///
/// The checkpoint callback receives the writable file and a label after the initial commit,
/// before each checkpoint close, and after reopening. The observer receives the closed file's
/// path and a label before each reopen and after the final close. The returned report describes
/// the final reopened state before its editing session closes.
///
/// # Panics
///
/// Panics if a file operation fails, a workload assertion fails, or either callback panics.
pub fn run_endurance_with_checkpoints(
    strategy: FileSpaceStrategy,
    observer: impl FnMut(&Path, &str),
    checkpoint: impl FnMut(&File, &str),
) -> String {
    run_endurance_inner(strategy, Some(observer), checkpoint)
}

fn run_endurance_inner(
    strategy: FileSpaceStrategy,
    mut observer: Option<impl FnMut(&Path, &str)>,
    mut checkpoint: impl FnMut(&File, &str),
) -> String {
    let path = temp::temp_path("free_space_endurance.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_strategy(strategy, strategy != FileSpaceStrategy::None, 1);
    if strategy == FileSpaceStrategy::Page {
        builder.with_file_space_page_size(FileSpacePageSize::try_from(4_096).unwrap());
    }
    builder.write(&path).unwrap();
    let mut file = open_writer(&path).unwrap();
    for sequence in 0..LIVE_GROUPS {
        GroupSpec::new(sequence).write(&Context::new(&file, &path, strategy, 0));
    }
    Context::new(&file, &path, strategy, 0).unwrap(file.commit());
    Context::new(&file, &path, strategy, 0).assert_size();
    checkpoint(&file, "initial-commit");
    let mut sizes = Vec::with_capacity(CYCLES);
    let mut recovered = Vec::new();
    let mut reused_after_reopen = false;
    for iteration in 1..=CYCLES {
        let context = Context::new(&file, &path, strategy, iteration);
        context.unwrap(file.root().delete(&GroupSpec::new(iteration - 1).path()));
        let spec = GroupSpec::new(LIVE_GROUPS + iteration - 1);
        spec.write(&context);
        context.unwrap(file.commit());
        let context = Context::new(&file, &path, strategy, iteration);
        sizes.push(context.physical);
        context.assert_size();
        if !recovered.is_empty() {
            reused_after_reopen |=
                file_space::raw_extents(&file, &format!("{}/contiguous", spec.path()))
                    .iter()
                    .any(|raw| {
                        recovered
                            .iter()
                            .any(|&(start, len)| raw.start < start + len && start < raw.end)
                    });
        }
        if iteration % REOPEN_INTERVAL != 0 {
            continue;
        }
        context.assert_survivors();
        if strategy != FileSpaceStrategy::None {
            if !recovered.is_empty() {
                assert!(
                    reused_after_reopen,
                    "persisted free extents must be reused; {context}"
                );
            }
        }
        checkpoint(&file, &format!("commit-{iteration}"));
        let before_close = context.to_string();
        file.close()
            .unwrap_or_else(|error| panic!("close: {error:?}; {before_close}"));
        let reader = File::open(&path)
            .unwrap_or_else(|error| panic!("reopen reader: {error:?}; {before_close}"));
        let context = Context::new(&reader, &path, strategy, iteration);
        context.assert_size();
        context.assert_survivors();
        recovered = context.unwrap(reader.persisted_free_space());
        if strategy != FileSpaceStrategy::None {
            assert!(!recovered.is_empty(), "free space must persist; {context}");
        }
        reader
            .close()
            .unwrap_or_else(|error| panic!("close reader: {error:?}; {before_close}"));
        file_space::assert_eof_matches_physical_size(&path);
        if let Some(observer) = &mut observer {
            observer(&path, &format!("reopen-{iteration}"));
        }
        file = open_writer(&path)
            .unwrap_or_else(|error| panic!("reopen writer: {error:?}; {before_close}"));
        let context = Context::new(&file, &path, strategy, iteration);
        if strategy != FileSpaceStrategy::None {
            assert_eq!(
                context.unwrap(file.space_accounting()).reusable_free_bytes,
                extent_bytes(&recovered),
                "reopening must recover every persisted free byte; {context}"
            );
        }
        checkpoint(&file, &format!("reopen-{iteration}"));
        reused_after_reopen = false;
    }
    let context = Context::new(&file, &path, strategy, CYCLES);
    assert_eq!(sizes.len(), CYCLES, "{context}");
    if strategy == FileSpaceStrategy::None {
        assert!(
            sizes[CYCLES - 1] > MAX_FILE_BYTES,
            "NONE must exceed the size limit; {context}"
        );
        assert!(
            sizes[1_023] >= sizes[511] + 256 * 1_024,
            "NONE must grow by at least 256 KiB after iteration 512 (size_512={}); {context}",
            sizes[511]
        );
    } else {
        let earlier_peak = sizes[256..512].iter().max().unwrap();
        let later_peak = sizes[768..1_024].iter().max().unwrap();
        assert!(
            *later_peak <= earlier_peak + 32 * 1_024,
            "peak size must plateau (iterations_257_512={earlier_peak}, \
             iterations_769_1024={later_peak}); {context}"
        );
    }
    context.assert_survivors();
    let report = context.final_characteristics(&recovered);
    let final_state = context.to_string();
    file.close()
        .unwrap_or_else(|error| panic!("final close: {error:?}; {final_state}"));
    if let Some(observer) = &mut observer {
        observer(&path, "final-close");
    }
    report
}

/// Assertion diagnostics for one file and iteration.
struct Context<'a> {
    file: &'a File,
    strategy: FileSpaceStrategy,
    iteration: usize,
    /// The physical file size in bytes when this context was created.
    physical: u64,
}

impl<'a> Context<'a> {
    fn new(file: &'a File, path: &Path, strategy: FileSpaceStrategy, iteration: usize) -> Self {
        Self {
            file,
            strategy,
            iteration,
            physical: fs::metadata(path).unwrap().len(),
        }
    }

    #[track_caller]
    fn unwrap<T>(&self, result: Result<T, impl Debug>) -> T {
        result.unwrap_or_else(|error| panic!("{error:?}; {self}"))
    }

    fn assert_size(&self) {
        if self.strategy != FileSpaceStrategy::None {
            assert!(
                self.physical <= MAX_FILE_BYTES,
                "file must stay within 1 MiB; {self}"
            );
        }
    }

    fn assert_survivors(&self) {
        let mut expected = (self.iteration..self.iteration + LIVE_GROUPS)
            .map(|sequence| GroupSpec::new(sequence).path())
            .collect::<Vec<_>>();
        expected.sort();
        let mut actual = self.unwrap(self.file.root().groups());
        actual.sort();
        assert_eq!(actual, expected, "live group names; {self}");
        for sequence in self.iteration..self.iteration + LIVE_GROUPS {
            GroupSpec::new(sequence).assert_data(self);
        }
        for sequence in 0..self.iteration {
            let path = GroupSpec::new(sequence).path();
            let error = match self.file.group(&path) {
                Ok(_) => panic!("deleted group {path:?} must be absent; {self}"),
                Err(error) => error,
            };
            let Error::Format(FormatError::PathNotFound(missing)) = &error else {
                panic!("expected PathNotFound, got {error:?}; {self}");
            };
            assert_eq!(missing, &path, "deleted group must be absent; {self}");
        }
    }

    /// Formats creation settings, file sizes, and reusable free space for the final snapshot.
    fn final_characteristics(&self, persisted: &[(u64, u64)]) -> String {
        let Self {
            file,
            strategy: _,
            iteration,
            physical,
        } = self;
        let FileSpaceInfo {
            strategy,
            persist,
            threshold,
            page_size,
            ..
        } = file
            .file_space_info()
            .unwrap_or_else(|| panic!("file-space creation settings must be present; {self}"));
        let accounting = self.unwrap(file.space_accounting());
        file_space::assert_space_accounting_consistent(&accounting);
        let mut out = String::new();
        writeln!(out, "small-file free-space endurance").unwrap();
        writeln!(out, "===============================").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "state: final reopen after {iteration} cycles").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "creation settings").unwrap();
        writeln!(out, "  strategy          : {strategy:?}").unwrap();
        writeln!(out, "  persistence       : {persist}").unwrap();
        writeln!(out, "  threshold (bytes) : {threshold}").unwrap();
        if *strategy == FileSpaceStrategy::Page {
            writeln!(out, "  page size (bytes) : {}", page_size.get()).unwrap();
        }
        writeln!(out).unwrap();
        writeln!(out, "file").unwrap();
        writeln!(out, "  physical size     : {physical} bytes").unwrap();
        writeln!(
            out,
            "  logical EOF       : {} bytes",
            accounting.logical_size
        )
        .unwrap();
        writeln!(
            out,
            "  live free bytes   : {}",
            accounting.reusable_free_bytes
        )
        .unwrap();
        writeln!(
            out,
            "  live free sections: {}",
            accounting.reusable_free_space.len()
        )
        .unwrap();
        writeln!(out, "  persisted sections: {}", persisted.len()).unwrap();
        writeln!(out, "  cycles            : {iteration}").unwrap();
        writeln!(
            out,
            "  top-level groups  : {}",
            self.unwrap(file.root().groups()).len()
        )
        .unwrap();
        writeln!(out).unwrap();
        writeln!(out, "accounting").unwrap();
        writeln!(
            out,
            "  persisted reusable free : {}",
            extent_bytes(persisted)
        )
        .unwrap();
        out
    }
}

impl Display for Context<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        let Self {
            file,
            strategy,
            iteration,
            physical,
        } = self;
        write!(
            formatter,
            "strategy={:?}, iteration={}, physical_size={}, reusable_free={:?}",
            strategy,
            iteration,
            physical,
            file.space_accounting()
                .map(|accounting| accounting.reusable_free_bytes)
                .or_else(|_| file.persisted_free_space().map(|free| extent_bytes(&free))),
        )
    }
}

/// The layout and expected contents of one group in the repeating workload.
struct GroupSpec {
    sequence: usize,
    pattern: usize,
}

impl GroupSpec {
    fn new(sequence: usize) -> Self {
        Self {
            sequence,
            pattern: sequence % LIVE_GROUPS,
        }
    }

    fn write(&self, context: &Context<'_>) {
        let &Self { sequence, pattern } = self;
        context.unwrap(
            context
                .file
                .root()
                .create_group_with(&self.path(), |group| {
                    group.set_attr("sequence", AttrValue::I64(i64::try_from(sequence).unwrap()));
                    for index in 0..self.attribute_count() {
                        group.set_attr(
                            &format!("a{index:02}"),
                            AttrValue::U8Array(self.attribute(index)),
                        );
                    }
                    group.create_dataset("contiguous", |builder| {
                        builder.with_i32_data(&self.data(0));
                        builder.set_attr("tag", AttrValue::U8Array(self.attribute(0)));
                    });
                    group.create_dataset("chunked", |builder| {
                        builder.with_i32_data(&self.data(1)).with_chunks(&[32]);
                    });
                    group.create_group_with("nested", |nested| {
                        nested.set_attr("tag", AttrValue::U8Array(self.attribute(1)));
                        for index in 0..1 + pattern % 3 {
                            nested.create_dataset(&format!("child{index}"), |builder| {
                                builder.with_i32_data(&self.child_data(index));
                            });
                        }
                    });
                }),
        );
    }

    fn assert_data(&self, context: &Context<'_>) {
        let &Self { sequence, pattern } = self;
        let group = context.unwrap(context.file.group(&self.path()));
        let mut expected = (0..self.attribute_count())
            .map(|index| {
                (
                    format!("a{index:02}"),
                    AttrValue::U8Array(self.attribute(index)),
                )
            })
            .collect::<HashMap<_, _>>();
        expected.insert(
            "sequence".into(),
            AttrValue::I64(i64::try_from(sequence).unwrap()),
        );
        assert_eq!(
            context.unwrap(group.attrs()),
            expected,
            "group attributes; {context}"
        );
        for (dataset_index, name) in ["contiguous", "chunked"].into_iter().enumerate() {
            let dataset = context.unwrap(group.dataset(name));
            assert_eq!(
                context.unwrap(dataset.read_i32()),
                self.data(dataset_index),
                "{name}; {context}"
            );
            let layout = context.unwrap(dataset.layout());
            match (dataset_index, layout) {
                (0, Layout::Contiguous { .. }) => {}
                (1, Layout::Chunked { chunk_shape, .. }) => {
                    assert_eq!(chunk_shape, [32], "{context}");
                }
                (_, layout) => panic!("unexpected {name} layout {layout:?}; {context}"),
            }
            if dataset_index == 0 {
                assert_eq!(
                    context.unwrap(dataset.attrs()),
                    [("tag".into(), AttrValue::U8Array(self.attribute(0)))].into(),
                    "dataset attributes; {context}"
                );
            }
        }
        let nested = context.unwrap(group.group("nested"));
        assert_eq!(
            context.unwrap(nested.attrs()),
            [("tag".into(), AttrValue::U8Array(self.attribute(1)))].into(),
            "nested attributes; {context}"
        );
        let mut children = context.unwrap(nested.datasets());
        children.sort();
        assert_eq!(
            children,
            (0..1 + pattern % 3)
                .map(|index| format!("child{index}"))
                .collect::<Vec<_>>(),
            "nested children; {context}"
        );
        for index in 0..1 + pattern % 3 {
            assert_eq!(
                context.unwrap(
                    context
                        .unwrap(nested.dataset(&format!("child{index}")))
                        .read_i32()
                ),
                self.child_data(index),
                "child data; {context}"
            );
        }
    }

    fn path(&self) -> String {
        format!("group{:06}", self.sequence)
    }

    fn attribute_count(&self) -> usize {
        if self.pattern % 4 == 0 {
            12
        } else {
            2 + self.pattern % 4
        }
    }

    fn attribute(&self, index: usize) -> Vec<u8> {
        let &Self { sequence, pattern } = self;
        (0..3 + (pattern * 7 + index * 3) % 29)
            .map(|offset| u8::try_from((sequence + index + offset) % 251).unwrap())
            .collect()
    }

    fn data(&self, dataset_index: usize) -> Vec<i32> {
        let &Self { sequence, pattern } = self;
        (0..64 * (1 + (pattern + dataset_index * 3) % 8))
            .map(|offset| {
                i32::try_from(sequence * 10_000 + dataset_index * 1_000 + offset).unwrap()
            })
            .collect()
    }

    fn child_data(&self, index: usize) -> Vec<i32> {
        (0..64)
            .map(|offset| i32::try_from(self.sequence * 10_000 + index * 100 + offset).unwrap())
            .collect()
    }
}

fn open_writer(path: &Path) -> Result<File, Error> {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_memory_strategy(MemoryStrategy::Bounded)
            .with_sync_policy(SyncPolicy::OnClose),
    )
}

fn extent_bytes(extents: &[(u64, u64)]) -> u64 {
    extents.iter().map(|&(_, len)| len).sum()
}

const CYCLES: usize = 1_024;
const LIVE_GROUPS: usize = 16;
const REOPEN_INTERVAL: usize = 64;
const MAX_FILE_BYTES: u64 = 1_024 * 1_024;
