//! Serialized timing samples, storage observations, and comparisons with matching 8/8 cases.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::Path;
use std::process::Command;

use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::Layout;
use hdf5_pure::LengthWidth;
use hdf5_pure::MetadataCacheStats;
use hdf5_pure::OffsetWidth;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::{Dataspace, DataspaceType};
use serde::Serialize;
use test_util_hdf5::__private::width_workload::{Case, Index, Observation, Phase};
use test_util_hdf5::file_space::persisted::PersistedStrategy;

/// Run configuration, operation samples, and per-phase storage observations.
#[derive(Serialize)]
pub(super) struct Report {
    configuration: serde_json::Value,
    /// Supported workload cases and their measurements.
    pub(super) rows: Vec<Row>,
    unsupported_unlimited_maximum: Vec<Label>,
}

impl Report {
    /// Creates an empty report with the run configuration and excluded cases.
    pub(super) fn new(samples: usize, cache_bytes: usize, unsupported: Vec<Label>) -> Self {
        let example_case = Case::new(
            OffsetWidth::Eight,
            LengthWidth::Eight,
            Index::Finite,
            PersistedStrategy::FsmAggr,
            false,
        );
        // Eight-byte lengths represent the unlimited marker, and a finite maximum also fits.
        let example_case =
            example_case.expect("default finite benchmark configuration is supported");
        Self {
            configuration: serde_json::json!({
                "schema_version": 1,
                "package_version": env!("CARGO_PKG_VERSION"),
                "samples_per_case": samples,
                "reproduction_command": format!("cargo run --locked --release --example width_benchmarks -- {samples} {cache_bytes}"),
                "cpu_control": "no affinity or frequency policy imposed",
                "warmup_rounds": 1,
                "metadata_cache_budget_bytes": cache_bytes,
                "edit_access_properties": format!("{:?}", example_case.access(cache_bytes)),
                "persistence": true,
                "free_section_threshold_bytes": 1,
                "configured_page_size_bytes": FileSpacePageSize::MIN.get(),
                "phase_order": ["create", "append", "reclaim", "reuse"],
                "timing_unit": "nanoseconds",
                "median": "upper middle sorted sample",
                "case_order": "rotate starting case and alternate traversal direction each round",
                "create_interval": "Case::create including input staging and closed FileBuilder::write; no explicit fsync",
                "append_interval": "both numeric append calls; opening, commit and close excluded",
                "commit_interval": "File::commit after staged scratch deletion and attribute replacement; staging and close excluded",
                "reopen_interval": "bounded editable open after completed reuse; verification and close excluded",
                "sync_policy": "OnClose; every close and its synchronization is outside timing intervals",
                "os_cache": "not flushed; files are written and validated before later operations",
                "cache_observation": "first verification/model/observation pass, then reset counters and repeat on the retained cache",
                "space_observation": "closed-file persisted free sections; file remainder is not live-only or field-only bytes",
                "wire_model": "serialized superblock, v2 dataset dataspace bodies reconstructed from verified shapes/maxima, and observed three-string VL descriptor storage; excludes other fields and padding; overlaps dataset_bytes",
                "workload": {"metadata_members":9,"metadata_attributes":9,"series_initial_elements":8,"series_append_elements":4,"series_chunk_elements":4,"finite_series_maximum":24,"single_initial_elements":1,"single_append_elements":1,"single_maximum":4,"strings":["alpha","","ζ"],"scratch_datasets":4,"scratch_bytes_each":128,"spacer_bytes_each":64},
                "debug_assertions": cfg!(debug_assertions),
                "library_features": {"std":cfg!(feature="std"),"checksum":cfg!(feature="checksum"),"deflate":cfg!(feature="deflate"),"fast-deflate":cfg!(feature="fast-deflate"),"ndarray":cfg!(feature="ndarray"),"num-complex":cfg!(feature="num-complex"),"serde":cfg!(feature="serde"),"provenance":cfg!(feature="provenance"),"zfp":cfg!(feature="zfp")},
                "environment": environment(),
            }),
            rows: Vec::new(),
            unsupported_unlimited_maximum: unsupported,
        }
    }

    /// Computes timing summaries and differences from matching 8/8 cases.
    ///
    /// # Panics
    ///
    /// Panics if a row has no 8/8 reference with the same index, strategy, and filter settings.
    pub(super) fn finish(&mut self) {
        let baseline = self
            .rows
            .iter()
            .filter(|row| row.case.offset_bytes == 8 && row.case.length_bytes == 8)
            .map(|row| {
                (
                    (row.case.index, row.case.strategy, row.case.filtered),
                    row.snapshots,
                )
            })
            .collect::<BTreeMap<_, _>>();
        for row in &mut self.rows {
            // Every supported index, strategy and filter factor includes an 8/8 case.
            let reference = baseline
                .get(&(row.case.index, row.case.strategy, row.case.filtered))
                .expect("matrix includes default-width baseline");
            row.finish(reference);
        }
    }
}

/// A width pair and the workload's index, file-space, and filter settings.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub(super) struct Label {
    offset_bytes: u8,
    length_bytes: u8,
    index: &'static str,
    strategy: &'static str,
    filtered: bool,
}

impl Label {
    /// Creates a label with byte widths and named index and strategy settings.
    pub(super) fn new(
        offsets: OffsetWidth,
        lengths: LengthWidth,
        index: Index,
        strategy: PersistedStrategy,
        filtered: bool,
    ) -> Self {
        Self {
            offset_bytes: offsets.get(),
            length_bytes: lengths.get(),
            index: match index {
                Index::Extensible => "extensible",
                Index::Finite => "finite",
            },
            strategy: match strategy {
                PersistedStrategy::FsmAggr => "fsm_aggr",
                PersistedStrategy::Page => "page",
            },
            filtered,
        }
    }
}

/// Timing samples and repeatable observations for one workload configuration.
#[derive(Serialize)]
pub(super) struct Row {
    /// Widths and workload settings shared by every sample.
    pub(super) case: Label,
    create_ns: Vec<u128>,
    append_ns: Vec<u128>,
    commit_ns: Vec<u128>,
    reopen_ns: Vec<u128>,
    snapshots: [Snapshot; 4],
    summaries: BTreeMap<&'static str, Summary>,
    differences_from_eight_eight: Vec<Difference>,
}

impl Row {
    /// Creates a row from its first measured sample.
    pub(super) fn new(case: Label, sample: Sample) -> Self {
        let Sample {
            create_ns,
            append_ns,
            commit_ns,
            reopen_ns,
            snapshots,
        } = sample;
        Self {
            case,
            create_ns: vec![create_ns],
            append_ns: vec![append_ns],
            commit_ns: vec![commit_ns],
            reopen_ns: vec![reopen_ns],
            snapshots,
            summaries: BTreeMap::new(),
            differences_from_eight_eight: Vec::new(),
        }
    }

    /// Adds timing values after checking that every snapshot matches the first sample.
    ///
    /// # Panics
    ///
    /// Panics if any storage, wire-model, or cache observation differs from the first sample.
    pub(super) fn push(&mut self, sample: Sample) {
        let Sample {
            create_ns,
            append_ns,
            commit_ns,
            reopen_ns,
            snapshots,
        } = sample;
        let Self {
            case: _,
            create_ns: creates,
            append_ns: appends,
            commit_ns: commits,
            reopen_ns: reopens,
            snapshots: previous,
            summaries: _,
            differences_from_eight_eight: _,
        } = self;
        assert_eq!(
            *previous, snapshots,
            "space/cache observations change within a benchmark configuration"
        );
        creates.push(create_ns);
        appends.push(append_ns);
        commits.push(commit_ns);
        reopens.push(reopen_ns);
    }

    fn finish(&mut self, reference: &[Snapshot; 4]) {
        let Self {
            case: _,
            create_ns,
            append_ns,
            commit_ns,
            reopen_ns,
            snapshots,
            summaries,
            differences_from_eight_eight,
        } = self;
        *summaries = [
            ("create", create_ns),
            ("append", append_ns),
            ("commit", commit_ns),
            ("reopen", reopen_ns),
        ]
        .into_iter()
        .map(|(name, values)| (name, Summary::new(values)))
        .collect();
        *differences_from_eight_eight = snapshots
            .iter()
            .zip(reference)
            .map(|(actual, reference)| Difference::new(actual, reference))
            .collect();
    }
}

/// One measured lifecycle with post-close observations for all four phases.
pub(super) struct Sample {
    /// Fixture construction and write interval, in nanoseconds.
    pub(super) create_ns: u128,
    /// Combined numeric append interval, in nanoseconds.
    pub(super) append_ns: u128,
    /// Reclamation commit interval, in nanoseconds.
    pub(super) commit_ns: u128,
    /// Bounded editable-open interval after reuse, in nanoseconds.
    pub(super) reopen_ns: u128,
    /// Create, append, reclaim, and reuse snapshots, in that order.
    pub(super) snapshots: [Snapshot; 4],
}

/// Extremes and an upper-middle median of nanosecond samples.
#[derive(Debug, Eq, PartialEq, Serialize)]
pub(super) struct Summary {
    minimum_ns: u128,
    median_ns: u128,
    maximum_ns: u128,
}

impl Summary {
    /// Computes the minimum, upper-middle median, and maximum of `values`.
    ///
    /// # Panics
    ///
    /// Panics if `values` is empty.
    pub(super) fn new(values: &[u128]) -> Self {
        let mut ordered = values.to_vec();
        ordered.sort_unstable();
        // Row::new inserts one sample before summaries are computed.
        let count = NonZeroUsize::new(ordered.len()).expect("each row contains a sample");
        Self {
            minimum_ns: ordered[0],
            median_ns: ordered[count.get() / 2],
            maximum_ns: ordered[count.get() - 1],
        }
    }
}

/// Space, selected wire sizes, and cache observations for one phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(super) struct Snapshot {
    space: Space,
    wire_model: Wire,
    first_pass_cache: Option<Cache>,
    second_pass_cache: Option<Cache>,
}

impl Snapshot {
    /// Verifies a read-only fixture and records two passes through its metadata-read cache.
    ///
    /// Resets counters between passes while retaining cached entries. Closes the file after
    /// collecting observations.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if opening, wire or storage measurement, or closing fails.
    ///
    /// # Panics
    ///
    /// Panics if fixture verification fails or the passes disagree on storage or wire sizes.
    pub(super) fn capture(
        case: Case,
        path: &Path,
        cache_bytes: usize,
        phase: Phase,
    ) -> Result<Self, Error> {
        let file = case.open_read(path, cache_bytes)?;
        case.verify(&file, phase);
        let wire_model = Wire::observe(case, &file)?;
        let first = case.observe(&file)?;
        file.reset_metadata_cache_stats();
        case.verify(&file, phase);
        let second_wire = Wire::observe(case, &file)?;
        let second = case.observe(&file)?;
        file.close()?;
        let snapshot = Self {
            space: Space::from(first),
            wire_model,
            first_pass_cache: first.metadata_cache.map(Cache::from),
            second_pass_cache: second.metadata_cache.map(Cache::from),
        };
        assert_same_snapshot(&snapshot, Space::from(second), second_wire);
        Ok(snapshot)
    }
}

fn assert_same_snapshot(first: &Snapshot, second_space: Space, second_wire: Wire) {
    assert_eq!(first.space, second_space);
    assert_eq!(first.wire_model, second_wire);
}

/// File-space byte totals and the reusable-section count.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct Space {
    file_bytes: u64,
    reusable_bytes: u64,
    non_reusable_bytes: u64,
    reusable_sections: usize,
    manager_bytes: u64,
    dataset_bytes: u64,
}

impl From<Observation> for Space {
    fn from(observed: Observation) -> Self {
        let Observation {
            file_bytes,
            reusable_bytes,
            non_reusable_bytes,
            reusable_sections,
            manager_bytes,
            dataset_bytes,
            metadata_cache: _,
        } = observed;
        Self {
            file_bytes,
            reusable_bytes,
            non_reusable_bytes,
            reusable_sections,
            manager_bytes,
            dataset_bytes,
        }
    }
}

/// Metadata-read-cache counters and retained payload capacity after one pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct Cache {
    bytes: usize,
    entries: usize,
    hits: u64,
    misses: u64,
    oversize_reads: u64,
    evictions: u64,
    invalidations: u64,
}

impl From<MetadataCacheStats> for Cache {
    fn from(stats: MetadataCacheStats) -> Self {
        Self {
            bytes: stats.bytes(),
            entries: stats.entries(),
            hits: stats.hits(),
            misses: stats.misses(),
            oversize_reads: stats.oversize_reads(),
            evictions: stats.evictions(),
            invalidations: stats.invalidations(),
        }
    }
}

/// Selected encoded sizes for the file-width comparison.
///
/// Includes the superblock, reconstructed dataset dataspace bodies, and variable-length
/// descriptors. Descriptor storage is also included in [`Space::dataset_bytes`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct Wire {
    superblock_bytes: u64,
    dataset_dataspace_bytes: u64,
    vl_descriptor_bytes: u64,
}

impl Wire {
    /// Measures selected encoded structures using the recorded file widths.
    ///
    /// Dataset dataspace bodies are reconstructed as version 2 messages from verified shapes
    /// and maxima. The dimension fields use the file length width, as defined in "The Dataspace
    /// Message" of the [format specification, version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a serialized size or the sum of encoded dataspace
    /// sizes exceeds [`u64::MAX`], a rank exceeds [`u8::MAX`], or string storage is not contiguous.
    /// Propagates metadata and serialization errors.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_simple
    fn observe(case: Case, file: &File) -> Result<Self, Error> {
        let superblock_bytes =
            u64::try_from(format::serialize_superblock(file.superblock())?.len())
                .map_err(|_error| Error::EditUnsupported("superblock length exceeds u64"))?;
        let mut dataset_dataspace_bytes = 0_u64;
        for group_name in ["metadata", "data", "holes"] {
            let group = file.group(group_name)?;
            for link_name in group.datasets()? {
                let dataset = group.dataset(&link_name)?;
                let dimensions = dataset.shape()?;
                let rank = u8::try_from(dimensions.len()).map_err(|_error| {
                    Error::EditUnsupported("benchmark dataspace rank exceeds u8")
                })?;
                let dataspace = Dataspace {
                    space_type: DataspaceType::Simple,
                    rank,
                    dimensions,
                    max_dimensions: dataset.maxshape()?,
                };
                let length = u64::try_from(dataspace.serialize(case.lengths().get())?.len())
                    .map_err(|_error| Error::EditUnsupported("dataspace length exceeds u64"))?;
                dataset_dataspace_bytes = dataset_dataspace_bytes
                    .checked_add(length)
                    .ok_or(Error::EditUnsupported("dataspace total exceeds u64"))?;
            }
        }
        let Layout::Contiguous {
            size: vl_descriptor_bytes,
            ..
        } = file.dataset("data/strings")?.layout()?
        else {
            return Err(Error::EditUnsupported(
                "benchmark strings are not contiguous",
            ));
        };
        Ok(Self {
            superblock_bytes,
            dataset_dataspace_bytes,
            vl_descriptor_bytes,
        })
    }
}

/// Signed per-phase differences from the matching 8/8 case.
///
/// Each value is the observed size minus the reference size. The selected wire model overlaps
/// dataset storage and covers only part of the encoded file.
#[derive(Serialize)]
struct Difference {
    file_bytes: i128,
    reusable_bytes: i128,
    manager_bytes: i128,
    dataset_bytes: i128,
    selected_wire_model_bytes: i128,
}

impl Difference {
    fn new(actual: &Snapshot, reference: &Snapshot) -> Self {
        let Space {
            file_bytes,
            reusable_bytes,
            non_reusable_bytes: _,
            reusable_sections: _,
            manager_bytes,
            dataset_bytes,
        } = actual.space;
        let Space {
            file_bytes: reference_file,
            reusable_bytes: reference_reusable,
            non_reusable_bytes: _,
            reusable_sections: _,
            manager_bytes: reference_managers,
            dataset_bytes: reference_datasets,
        } = reference.space;
        let wire = |value: Wire| {
            let Wire {
                superblock_bytes,
                dataset_dataspace_bytes,
                vl_descriptor_bytes,
            } = value;
            i128::from(superblock_bytes)
                + i128::from(dataset_dataspace_bytes)
                + i128::from(vl_descriptor_bytes)
        };
        Self {
            file_bytes: i128::from(file_bytes) - i128::from(reference_file),
            reusable_bytes: i128::from(reusable_bytes) - i128::from(reference_reusable),
            manager_bytes: i128::from(manager_bytes) - i128::from(reference_managers),
            dataset_bytes: i128::from(dataset_bytes) - i128::from(reference_datasets),
            selected_wire_model_bytes: wire(actual.wire_model) - wire(reference.wire_model),
        }
    }
}

fn environment() -> BTreeMap<&'static str, String> {
    let mut values = BTreeMap::from([
        ("architecture", std::env::consts::ARCH.to_owned()),
        ("operating_system", std::env::consts::OS.to_owned()),
        (
            "temporary_directory",
            std::env::temp_dir().display().to_string(),
        ),
    ]);
    for (name, program, args) in [
        ("runtime_rustc", "rustc", vec!["-vV"]),
        ("system", "uname", vec!["-srmo"]),
        ("producer_revision", "git", vec!["rev-parse", "HEAD"]),
        (
            "producer_status",
            "git",
            vec!["status", "--porcelain", "--untracked-files=no"],
        ),
        ("filesystem", "stat", vec!["-f", "-c", "%T"]),
    ] {
        let mut command = Command::new(program);
        command.args(args);
        if name == "filesystem" {
            command.arg(std::env::temp_dir());
        }
        if let Ok(output) = command.output()
            && output.status.success()
        {
            values.insert(
                name,
                String::from_utf8_lossy(&output.stdout).trim().to_owned(),
            );
        }
    }
    if let Ok(cpu) = std::fs::read_to_string("/proc/cpuinfo")
        && let Some(model) = cpu.lines().find_map(|line| {
            line.strip_prefix("model name")
                .and_then(|line| line.split_once(':').map(|(_, value)| value.trim()))
        })
    {
        values.insert("cpu", model.to_owned());
    }
    values
}

#[cfg(test)]
mod tests {
    use hdf5_pure::{LengthWidth, OffsetWidth};
    use test_util_hdf5::__private::width_workload::{Case, Index};
    use test_util_hdf5::file_space::persisted::PersistedStrategy;

    use super::{Label, Row};

    #[test]
    #[should_panic(expected = "space/cache observations change within a benchmark configuration")]
    fn repeated_observation_control_detects_changed_file_size() {
        let case = Case::new(
            OffsetWidth::Eight,
            LengthWidth::Eight,
            Index::Finite,
            PersistedStrategy::FsmAggr,
            false,
        )
        .unwrap();
        let label = Label::new(
            OffsetWidth::Eight,
            LengthWidth::Eight,
            Index::Finite,
            PersistedStrategy::FsmAggr,
            false,
        );
        let mut row = Row::new(label, crate::sample(case, 65_536).unwrap());
        let mut changed = crate::sample(case, 65_536).unwrap();
        changed.snapshots[0].space.file_bytes += 1;
        row.push(changed);
    }
}
