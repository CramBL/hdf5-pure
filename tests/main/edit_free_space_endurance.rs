//! Tests free-space reuse across commits and repeated file opens, with NONE as a growth control.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fmt::{self, Debug, Display, Formatter};
use std::fs;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;

use hdf5_pure::AttrValue;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpaceInfo;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FormatError;
use hdf5_pure::Layout;
use hdf5_pure::SpaceOwnership;
use rstest::rstest;
use test_util::temp;
use test_util_hdf5::file_space;
use test_util_hdf5::session;

#[rstest]
#[case::fsm_aggr(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
#[case::none(FileSpaceStrategy::None)]
fn small_file_churn_plateaus_only_with_persistent_free_space(#[case] strategy: FileSpaceStrategy) {
    let path = temp::temp_path("free_space_endurance.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_strategy(strategy, strategy != FileSpaceStrategy::None, 1);
    if strategy == FileSpaceStrategy::Page {
        builder.with_file_space_page_size(FileSpacePageSize::try_from(4_096).unwrap());
    }
    builder.write(&path).unwrap();
    let mut file = session::open_bounded_on_close(&path).unwrap();
    for sequence in 0..LIVE_GROUPS {
        GroupSpec::new(sequence).write(&Context::new(&file, &path, strategy, 0));
    }
    Context::new(&file, &path, strategy, 0).unwrap(file.commit());
    Context::new(&file, &path, strategy, 0).assert_size();
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
            context.assert_exact_ownership();
            if !recovered.is_empty() {
                assert!(
                    reused_after_reopen,
                    "persisted free extents must be reused; {context}"
                );
            }
        }
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
        file = session::open_bounded_on_close(&path)
            .unwrap_or_else(|error| panic!("reopen writer: {error:?}; {before_close}"));
        let context = Context::new(&file, &path, strategy, iteration);
        if strategy != FileSpaceStrategy::None {
            assert_eq!(
                context.unwrap(file.space_accounting()).reusable_free_bytes,
                extent_bytes(&recovered),
                "reopening must recover every persisted free byte; {context}"
            );
            context.assert_exact_ownership();
        }
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
    insta::assert_snapshot!(
        format!("free_space_endurance_{strategy:?}").to_ascii_lowercase(),
        context.final_characteristics(extent_bytes(&recovered)),
    );
    let final_state = context.to_string();
    file.close()
        .unwrap_or_else(|error| panic!("final close: {error:?}; {final_state}"));
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

    fn assert_exact_ownership(&self) {
        panic::catch_unwind(AssertUnwindSafe(|| {
            file_space::assert_exact_space_ownership(self.file)
        }))
        .unwrap_or_else(|_| panic!("exact ownership failed; {self}"));
    }

    /// Formats creation settings and storage ownership for the final snapshot.
    fn final_characteristics(&self, persisted_free_bytes: u64) -> String {
        let Self {
            file,
            strategy: _,
            iteration,
            physical: _,
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
        let SpaceOwnership {
            physical_size,
            logical_size,
            userblock_space,
            live_raw_space,
            live_metadata_space,
            free_space_manager_space,
            reusable_free_space,
            unclassified_space,
            analysis_complete,
            trailing_slack_space,
        } = self.unwrap(file.__space_ownership());
        let categories = [
            ("userblock", &userblock_space),
            ("live raw data", &live_raw_space),
            ("live metadata", &live_metadata_space),
            ("free-space managers", &free_space_manager_space),
            ("reusable free space", &reusable_free_space),
            ("unclassified space", &unclassified_space),
            ("trailing slack", &trailing_slack_space),
        ];
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
        writeln!(out, "  physical size     : {physical_size} bytes").unwrap();
        writeln!(out, "  logical EOF       : {logical_size} bytes").unwrap();
        writeln!(
            out,
            "  top-level groups  : {}",
            self.unwrap(file.root().groups()).len()
        )
        .unwrap();
        writeln!(out, "  analysis complete : {analysis_complete}").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "storage ownership").unwrap();
        writeln!(out, "  {:<21} {:>10} {:>8}", "category", "bytes", "extents").unwrap();
        for (label, extents) in categories {
            writeln!(
                out,
                "  {label:<21} {:>10} {:>8}",
                extent_bytes(extents),
                extents.len()
            )
            .unwrap();
        }
        writeln!(out).unwrap();
        writeln!(out, "accounting").unwrap();
        writeln!(
            out,
            "  classified bytes        : {}",
            extent_bytes(&userblock_space)
                + extent_bytes(&live_raw_space)
                + extent_bytes(&live_metadata_space)
                + extent_bytes(&free_space_manager_space)
                + extent_bytes(&reusable_free_space)
        )
        .unwrap();
        writeln!(out, "  persisted reusable free : {persisted_free_bytes}").unwrap();
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
        )?;
        match file.__space_ownership() {
            Ok(SpaceOwnership {
                physical_size: _,
                logical_size,
                userblock_space: _,
                live_raw_space,
                live_metadata_space,
                free_space_manager_space,
                reusable_free_space: _,
                unclassified_space,
                analysis_complete,
                trailing_slack_space,
            }) => write!(
                formatter,
                ", analysis_complete={}, logical_size={}, live_raw={}, live_metadata={}, \
                 manager_bytes={}, unclassified_bytes={}, trailing_slack_bytes={}",
                analysis_complete,
                logical_size,
                extent_bytes(&live_raw_space),
                extent_bytes(&live_metadata_space),
                extent_bytes(&free_space_manager_space),
                extent_bytes(&unclassified_space),
                extent_bytes(&trailing_slack_space),
            ),
            Err(error) => write!(formatter, ", ownership_error={error:?}"),
        }
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

fn extent_bytes(extents: &[(u64, u64)]) -> u64 {
    extents.iter().map(|&(_, len)| len).sum()
}

const CYCLES: usize = 1_024;
const LIVE_GROUPS: usize = 16;
const REOPEN_INTERVAL: usize = 64;
const MAX_FILE_BYTES: u64 = 1_024 * 1_024;
