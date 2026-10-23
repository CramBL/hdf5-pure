//! Exercises free-space reuse with mixed raw data and metadata across editor lifecycles.
//!
//! The workload deletes the oldest groups under allocation pressure and verifies surviving data,
//! deleted paths, bounded growth, and Rust ownership accounting.

use std::collections::{BTreeSet, VecDeque};
use std::fmt::Write as _;
use std::path::Path;

use hdf5_pure::AttrValue;
use hdf5_pure::ChunkIndex;
use hdf5_pure::DeflateLevel;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FilterId;
use hdf5_pure::FormatError;
use hdf5_pure::Layout;
use hdf5_pure::MaxExtent;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::ScaleOffset;
use hdf5_pure::SpaceAccounting;
use hdf5_pure::UserblockSize;
use test_util::temp;

use crate::dataset::{self, Filter};
use crate::file_space;
use crate::file_space::SpaceOwnershipSummary;
use crate::session;
use crate::userblock;

/// Runs the churn workload and returns its growth, reuse, and ownership report.
///
/// # Panics
///
/// Panics if a file operation fails or a workload assertion fails.
pub fn run_churn(lifecycle: Lifecycle, strategy: Strategy, base: Base) -> ChurnReport {
    run_churn_inner(lifecycle, strategy, base, None::<fn(&Path, &str)>)
}

/// Runs churn with observations after each intended close and before the next writing session.
///
/// The observer receives the closed file's path and a checkpoint label. For
/// [`Lifecycle::ContinuousOpen`], it runs only after the final close.
///
/// # Panics
///
/// Panics if a file operation fails, a workload assertion fails, or the observer panics.
pub fn run_churn_observed(
    lifecycle: Lifecycle,
    strategy: Strategy,
    base: Base,
    observer: impl FnMut(&Path, &str),
) -> ChurnReport {
    run_churn_inner(lifecycle, strategy, base, Some(observer))
}

fn run_churn_inner(
    lifecycle: Lifecycle,
    strategy: Strategy,
    base: Base,
    mut observer: Option<impl FnMut(&Path, &str)>,
) -> ChurnReport {
    let path = temp::temp_path(&format!(
        "hdf5_pure_free_space_churn_{}_{}_{}.h5",
        lifecycle.label(),
        strategy.label(),
        base.label()
    ));
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("sentinel")
        .with_i32_data(&[7, 11, 13, 17]);
    strategy.configure(&mut builder);
    let stamped = base.write(&path, builder);
    assert!(base.image_size(&path) <= MAX_FILE_BYTES);

    let mut session = Some(base.open(&path));
    let mut writable_opens = 1usize;
    let mut closes = 0usize;
    let mut reopens = 0usize;
    let mut live = VecDeque::new();
    let mut created_names = BTreeSet::new();
    let mut deletion_waves = Vec::new();
    let mut reopen_checkpoints = Vec::new();
    let mut epochs_written = 0usize;
    let mut epochs_deleted = 0usize;
    let mut first_delete_after = None;
    let mut churn_epochs_after_first_delete = 0usize;
    let mut stats = RunStats {
        peak_physical_size: base.image_size(&path),
        peak_logical_size: 0,
        peak_pressure: 0,
        peak_reusable_free: 0,
        epoch_writes_growing_physical_file: 0,
        epoch_writes_without_physical_growth: 0,
        post_churn_epoch_writes_without_physical_growth: 0,
        reuse_write_observed: false,
        deletion_reduced_pressure_or_created_free: false,
    };

    while first_delete_after.is_none()
        || churn_epochs_after_first_delete < CHURN_EPOCHS_AFTER_FIRST_DELETE
    {
        let sequence = epochs_written;
        let spec = EpochSpec::new(sequence);
        assert!(
            created_names.insert(spec.name()),
            "epoch names must be unique"
        );

        let file = session.as_ref().unwrap();
        let before_acct = file.space_accounting().unwrap();
        let before_pressure = pressure(base, &before_acct);
        let before_physical = base.image_size(&path);

        write_epoch(file, spec);
        file.commit().unwrap();
        live.push_back(sequence);
        epochs_written += 1;
        if first_delete_after.is_some() {
            churn_epochs_after_first_delete += 1;
        }

        let mut sample = collect_sample(
            &path,
            base,
            file,
            sequence,
            live.len(),
            epochs_written,
            epochs_deleted,
        );
        stats.observe_epoch(sample);
        if sample.physical_size > before_physical {
            stats.epoch_writes_growing_physical_file += 1;
        } else {
            stats.epoch_writes_without_physical_growth += 1;
            if first_delete_after.is_some() {
                stats.post_churn_epoch_writes_without_physical_growth += 1;
            }
        }
        if first_delete_after.is_some()
            && sample.physical_size <= before_physical
            && sample.reusable_free < before_acct.reusable_free_bytes
            && sample.pressure > before_pressure
        {
            stats.reuse_write_observed = true;
        }

        if sample.pressure > HIGH_WATER_BYTES {
            let wave_before = sample;
            let active_before = live.len();
            let mut deleted = 0usize;
            while sample.pressure > LOW_WATER_BYTES {
                let oldest = live.pop_front().expect("high pressure implies live epochs");
                assert_eq!(
                    oldest, epochs_deleted,
                    "deletion must remain FIFO from the oldest live epoch"
                );
                file.root().delete(&EpochSpec::new(oldest).name()).unwrap();
                file.commit().unwrap();
                epochs_deleted += 1;
                deleted += 1;
                if first_delete_after.is_none() {
                    first_delete_after = Some(epochs_written);
                }
                sample = collect_sample(
                    &path,
                    base,
                    file,
                    sequence,
                    live.len(),
                    epochs_written,
                    epochs_deleted,
                );
                stats.observe_state(
                    sample.physical_size,
                    sample.logical_size,
                    sample.reusable_free,
                    sample.pressure,
                );
            }
            assert!(deleted > 0);
            assert!(
                sample.reusable_free > wave_before.reusable_free
                    || sample.pressure < wave_before.pressure,
                "an interior deletion wave must create reusable space or reduce pressure"
            );
            stats.deletion_reduced_pressure_or_created_free = true;
            let ownership = base.exact_ownership(file);
            deletion_waves.push(DeletionWave {
                index: deletion_waves.len() + 1,
                after_epoch: sequence,
                active_before,
                deleted,
                active_after: live.len(),
                physical_size: sample.physical_size,
                logical_size: sample.logical_size,
                reusable_free: sample.reusable_free,
                pressure: sample.pressure,
                ownership,
            });

            if lifecycle == Lifecycle::ReopenFrequently {
                reopen_session(
                    &path,
                    base,
                    &mut session,
                    ReopenReason::DeletionWave,
                    sequence,
                    &live,
                    epochs_written,
                    &mut reopen_checkpoints,
                    &mut writable_opens,
                    &mut closes,
                    &mut reopens,
                    &mut stats,
                    &mut observer,
                );
            }
        } else if lifecycle == Lifecycle::ReopenFrequently
            && first_delete_after.is_none()
            && epochs_written % INITIAL_REOPEN_INTERVAL == 0
        {
            reopen_session(
                &path,
                base,
                &mut session,
                ReopenReason::InitialFill,
                sequence,
                &live,
                epochs_written,
                &mut reopen_checkpoints,
                &mut writable_opens,
                &mut closes,
                &mut reopens,
                &mut stats,
                &mut observer,
            );
        }
    }

    let epochs_before_first_delete = first_delete_after.expect("churn must reach a deletion wave");
    assert!(churn_epochs_after_first_delete >= CHURN_EPOCHS_AFTER_FIRST_DELETE);
    assert!(
        deletion_waves.len() > 1,
        "sustained churn must create multiple deletion waves"
    );
    let max_epochs_deleted_in_one_wave = deletion_waves
        .iter()
        .map(|wave| wave.deleted)
        .max()
        .unwrap();
    assert!(
        max_epochs_deleted_in_one_wave > 1,
        "low-water hysteresis must produce a multi-epoch deletion wave"
    );

    if lifecycle == Lifecycle::ReopenBeforeFinalCheck {
        reopen_session(
            &path,
            base,
            &mut session,
            ReopenReason::FinalCheck,
            epochs_written - 1,
            &live,
            epochs_written,
            &mut reopen_checkpoints,
            &mut writable_opens,
            &mut closes,
            &mut reopens,
            &mut stats,
            &mut observer,
        );
    }

    // ContinuousOpen validates all epochs before its editing session closes.
    let file = session.as_ref().unwrap();
    let coverage = validate_all_epochs(file, &live, epochs_written);
    assert_eq!(
        file.dataset("sentinel").unwrap().read_i32().unwrap(),
        vec![7, 11, 13, 17]
    );
    let final_state = final_state(&path, base, file, &live);
    stats.observe_state(
        final_state.physical_size,
        final_state.logical_size,
        final_state.reusable_free,
        final_state.pressure,
    );

    let closing = session.take().unwrap();
    let after_close = closing.clone();
    closing.close().unwrap();
    closes += 1;
    let after_close_ownership = base.exact_ownership(&after_close);
    drop(after_close);
    if let Some(stamped) = stamped {
        stamped.assert_unchanged(&path);
    }
    let released_physical_size = base.image_size(&path);
    assert!(released_physical_size <= MAX_FILE_BYTES);
    stats.peak_physical_size = stats.peak_physical_size.max(released_physical_size);
    file_space::assert_eof_matches_physical_size(&path);
    let reader = File::open(&path).unwrap();
    assert_eq!(
        reader.dataset("sentinel").unwrap().read_i32().unwrap(),
        vec![7, 11, 13, 17]
    );
    let after_close_coverage = validate_all_epochs(&reader, &live, epochs_written);
    assert_eq!(after_close_coverage, coverage);
    reader.close().unwrap();
    if let Some(observer) = &mut observer {
        observer(&path, "final-close");
    }

    let RunStats {
        peak_physical_size,
        peak_logical_size,
        peak_pressure,
        peak_reusable_free,
        epoch_writes_growing_physical_file,
        epoch_writes_without_physical_growth,
        post_churn_epoch_writes_without_physical_growth,
        reuse_write_observed,
        deletion_reduced_pressure_or_created_free,
    } = stats;
    assert!(deletion_reduced_pressure_or_created_free);
    assert!(
        reuse_write_observed,
        "a post-delete epoch must increase allocation pressure without growing the physical file"
    );
    assert!(
        post_churn_epoch_writes_without_physical_growth > 0,
        "at least one epoch write after churn begins must avoid physical growth"
    );

    ChurnReport {
        lifecycle,
        epochs_written,
        epochs_deleted,
        epochs_live: live.len(),
        epochs_before_first_delete,
        churn_epochs_after_first_delete,
        writable_opens,
        closes,
        reopens,
        peak_physical_size,
        peak_logical_size,
        peak_pressure,
        peak_reusable_free,
        epoch_writes_growing_physical_file,
        epoch_writes_without_physical_growth,
        post_churn_epoch_writes_without_physical_growth,
        max_epochs_deleted_in_one_wave,
        deletion_waves,
        reopen_checkpoints,
        coverage,
        final_state,
        after_close_ownership,
    }
}

/// The allocation, lifecycle, coverage, and ownership observations from a churn run.
pub struct ChurnReport {
    lifecycle: Lifecycle,
    epochs_written: usize,
    epochs_deleted: usize,
    epochs_live: usize,
    epochs_before_first_delete: usize,
    churn_epochs_after_first_delete: usize,
    writable_opens: usize,
    closes: usize,
    reopens: usize,
    peak_physical_size: u64,
    peak_logical_size: u64,
    peak_pressure: u64,
    peak_reusable_free: u64,
    epoch_writes_growing_physical_file: usize,
    epoch_writes_without_physical_growth: usize,
    post_churn_epoch_writes_without_physical_growth: usize,
    max_epochs_deleted_in_one_wave: usize,
    deletion_waves: Vec<DeletionWave>,
    reopen_checkpoints: Vec<ReopenCheckpoint>,
    coverage: Coverage,
    final_state: FinalState,
    after_close_ownership: SpaceOwnershipSummary,
}

impl ChurnReport {
    /// Formats workload limits, deletion waves, reopen checkpoints, and ownership totals.
    pub fn render(&self) -> String {
        let ChurnReport {
            lifecycle,
            epochs_written,
            epochs_deleted,
            epochs_live,
            epochs_before_first_delete,
            churn_epochs_after_first_delete,
            writable_opens,
            closes,
            reopens,
            peak_physical_size,
            peak_logical_size,
            peak_pressure,
            peak_reusable_free,
            epoch_writes_growing_physical_file,
            epoch_writes_without_physical_growth,
            post_churn_epoch_writes_without_physical_growth,
            max_epochs_deleted_in_one_wave,
            deletion_waves,
            reopen_checkpoints,
            coverage,
            final_state,
            after_close_ownership,
        } = self;
        let FinalState {
            physical_size,
            logical_size,
            reusable_free,
            reusable_free_regions,
            pressure,
            live_epochs,
            ownership,
        } = final_state;
        let Coverage {
            nested_epoch_groups,
            compact_attr_objects,
            dense_attr_objects,
            compact_link_groups,
            high_fanout_compact_link_groups,
            dense_link_groups,
            single_chunk,
            fixed_array,
            extensible_array,
            rank1_datasets,
            rank2_datasets,
            aligned_chunk_grids,
            partial_edge_chunk_grids,
            filter_none,
            filter_deflate,
            filter_shuffle_deflate,
            filter_scale_offset,
            compressible_filtered,
            incompressible_filtered,
        } = coverage;
        let mut out = String::new();
        writeln!(out, "free-space churn").unwrap();
        writeln!(out, "================").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "scenario: {}", lifecycle.label()).unwrap();
        writeln!(out).unwrap();
        writeln!(out, "limits").unwrap();
        writeln!(out, "  maximum file size : {MAX_FILE_BYTES}").unwrap();
        writeln!(out, "  high-water        : {HIGH_WATER_BYTES}").unwrap();
        writeln!(out, "  low-water         : {LOW_WATER_BYTES}").unwrap();
        writeln!(out, "  headroom          : {HEADROOM_BYTES}").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "workload").unwrap();
        writeln!(out, "  epochs written             : {epochs_written}").unwrap();
        writeln!(out, "  epochs deleted             : {epochs_deleted}").unwrap();
        writeln!(out, "  epochs still live          : {epochs_live}").unwrap();
        writeln!(
            out,
            "  epochs before first delete : {epochs_before_first_delete}"
        )
        .unwrap();
        writeln!(
            out,
            "  churn epochs               : {churn_epochs_after_first_delete}"
        )
        .unwrap();
        writeln!(
            out,
            "  deletion waves             : {}",
            deletion_waves.len()
        )
        .unwrap();
        writeln!(
            out,
            "  max deleted in one wave    : {max_epochs_deleted_in_one_wave}"
        )
        .unwrap();
        writeln!(out).unwrap();
        writeln!(out, "lifecycle").unwrap();
        writeln!(out, "  writable opens : {writable_opens}").unwrap();
        writeln!(out, "  closes         : {closes}").unwrap();
        writeln!(out, "  reopens        : {reopens}").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "high water").unwrap();
        writeln!(out, "  physical file : {peak_physical_size}").unwrap();
        writeln!(out, "  logical file  : {peak_logical_size}").unwrap();
        writeln!(out, "  pressure      : {peak_pressure}").unwrap();
        writeln!(out, "  reusable free : {peak_reusable_free}").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "reuse").unwrap();
        writeln!(
            out,
            "  epoch writes growing physical file   : {epoch_writes_growing_physical_file}"
        )
        .unwrap();
        writeln!(
            out,
            "  epoch writes without physical growth : {epoch_writes_without_physical_growth}"
        )
        .unwrap();
        writeln!(
            out,
            "  post-churn writes without growth      : \
             {post_churn_epoch_writes_without_physical_growth}"
        )
        .unwrap();
        writeln!(out).unwrap();
        writeln!(out, "deletion waves").unwrap();
        writeln!(
            out,
            "  wave  after-epoch  live-before  deleted  live-after  physical  logical  reusable \
             live-raw  live-meta  unclassified  pressure"
        )
        .unwrap();
        for wave in deletion_waves {
            let DeletionWave {
                index,
                after_epoch,
                active_before,
                deleted,
                active_after,
                physical_size,
                logical_size,
                reusable_free,
                pressure,
                ownership,
            } = wave;

            writeln!(
                out,
                "  {index:>4}  {after_epoch:>11}  {active_before:>11}  {deleted:>7}  \
                 {active_after:>10}  {physical_size:>8}  {logical_size:>7}  {reusable_free:>8}  \
                 {:>8}  {:>9}  {:>8}  {pressure:>8}",
                ownership.live_raw_bytes,
                ownership.live_metadata_bytes,
                ownership.unclassified_bytes,
            )
            .unwrap();
        }
        writeln!(out).unwrap();
        writeln!(out, "reopen checkpoints").unwrap();
        writeln!(
            out,
            "  #  reason          after-epoch  physical  persisted-free  recovered-free  unclassified"
        )
        .unwrap();
        if reopen_checkpoints.is_empty() {
            writeln!(out, "  none").unwrap();
        } else {
            for checkpoint in reopen_checkpoints {
                let ReopenCheckpoint {
                    index,
                    reason,
                    after_epoch,
                    physical_size,
                    persisted_free,
                    recovered_free,
                    ownership,
                } = checkpoint;

                writeln!(
                    out,
                    "  {index:>1}  {:<14}  {after_epoch:>11}  {physical_size:>8}  \
                     {persisted_free:>14}  {recovered_free:>14}  {:>8}",
                    reason.label(),
                    ownership.unclassified_bytes,
                )
                .unwrap();
            }
        }
        writeln!(out).unwrap();
        writeln!(out, "final").unwrap();
        writeln!(out, "  physical file size         : {physical_size}").unwrap();
        writeln!(out, "  logical size               : {logical_size}").unwrap();
        writeln!(out, "  reusable free              : {reusable_free}").unwrap();
        writeln!(
            out,
            "  reusable free regions      : {reusable_free_regions}"
        )
        .unwrap();
        writeln!(out, "  pressure                   : {pressure}").unwrap();
        writeln!(out, "  live epochs                : {live_epochs}").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "coverage").unwrap();
        writeln!(out, "  object/group layout").unwrap();
        writeln!(
            out,
            "    nested epoch groups              : {nested_epoch_groups}"
        )
        .unwrap();
        writeln!(
            out,
            "    compact attribute objects        : {compact_attr_objects}"
        )
        .unwrap();
        writeln!(
            out,
            "    dense attribute objects          : {dense_attr_objects}"
        )
        .unwrap();
        writeln!(
            out,
            "    compact-link groups              : {compact_link_groups}"
        )
        .unwrap();
        writeln!(
            out,
            "    high-fanout compact-link groups  : {high_fanout_compact_link_groups}"
        )
        .unwrap();
        writeln!(
            out,
            "    dense-link groups                : {dense_link_groups} (writer unsupported)"
        )
        .unwrap();
        writeln!(out, "  chunk indices").unwrap();
        writeln!(out, "    single chunk                     : {single_chunk}").unwrap();
        writeln!(out, "    fixed array                      : {fixed_array}").unwrap();
        writeln!(
            out,
            "    extensible array                 : {extensible_array}"
        )
        .unwrap();
        writeln!(out, "  ranks").unwrap();
        writeln!(
            out,
            "    rank 1 datasets                  : {rank1_datasets}"
        )
        .unwrap();
        writeln!(
            out,
            "    rank 2 datasets                  : {rank2_datasets}"
        )
        .unwrap();
        writeln!(out, "  chunk grids").unwrap();
        writeln!(
            out,
            "    aligned                          : {aligned_chunk_grids}"
        )
        .unwrap();
        writeln!(
            out,
            "    partial-edge                     : {partial_edge_chunk_grids}"
        )
        .unwrap();
        writeln!(out, "  filters").unwrap();
        writeln!(out, "    none                             : {filter_none}").unwrap();
        writeln!(
            out,
            "    deflate                          : {filter_deflate}"
        )
        .unwrap();
        writeln!(
            out,
            "    shuffle + deflate                : {filter_shuffle_deflate}"
        )
        .unwrap();
        writeln!(
            out,
            "    integer scale-offset             : {filter_scale_offset}"
        )
        .unwrap();
        writeln!(out, "  filtered contents").unwrap();
        writeln!(
            out,
            "    compressible                     : {compressible_filtered}"
        )
        .unwrap();
        writeln!(
            out,
            "    incompressible                   : {incompressible_filtered}"
        )
        .unwrap();
        writeln!(out).unwrap();
        render_ownership(&mut out, "ownership while writer open", *ownership);
        writeln!(out).unwrap();
        render_ownership(
            &mut out,
            "ownership after writer close",
            *after_close_ownership,
        );
        out
    }
}

fn render_ownership(
    out: &mut String,
    heading: &str,
    SpaceOwnershipSummary {
        physical_size,
        logical_size,
        userblock_bytes: _,
        live_raw_bytes,
        live_metadata_bytes,
        free_space_manager_bytes,
        reusable_free_bytes,
        unclassified_bytes,
        trailing_slack_bytes,
        live_raw_extents,
        live_metadata_extents,
        free_space_manager_extents,
        reusable_free_extents,
        unclassified_extents,
        trailing_slack_extents,
    }: SpaceOwnershipSummary,
) {
    writeln!(out, "{heading}").unwrap();
    writeln!(out, "  physical file size       : {physical_size}").unwrap();
    writeln!(out, "  logical EOF              : {logical_size}").unwrap();
    writeln!(out, "  live raw bytes           : {live_raw_bytes}").unwrap();
    writeln!(out, "  live metadata bytes      : {live_metadata_bytes}").unwrap();
    writeln!(
        out,
        "  free-space-manager bytes : {free_space_manager_bytes}"
    )
    .unwrap();
    writeln!(out, "  reusable free bytes      : {reusable_free_bytes}").unwrap();
    writeln!(out, "  unclassified bytes       : {unclassified_bytes}").unwrap();
    writeln!(out, "  trailing slack bytes     : {trailing_slack_bytes}").unwrap();
    writeln!(out, "  live raw extents         : {live_raw_extents}").unwrap();
    writeln!(out, "  live metadata extents    : {live_metadata_extents}").unwrap();
    writeln!(
        out,
        "  manager metadata extents : {free_space_manager_extents}"
    )
    .unwrap();
    writeln!(out, "  reusable free extents    : {reusable_free_extents}").unwrap();
    writeln!(out, "  unclassified extents     : {unclassified_extents}").unwrap();
    writeln!(out, "  trailing slack extents   : {trailing_slack_extents}").unwrap();
    writeln!(out, "  exact partition          : yes").unwrap();
}

struct RunStats {
    peak_physical_size: u64,
    peak_logical_size: u64,
    peak_pressure: u64,
    peak_reusable_free: u64,
    epoch_writes_growing_physical_file: usize,
    epoch_writes_without_physical_growth: usize,
    post_churn_epoch_writes_without_physical_growth: usize,
    reuse_write_observed: bool,
    deletion_reduced_pressure_or_created_free: bool,
}

impl RunStats {
    fn observe_epoch(
        &mut self,
        ChurnSample {
            sequence,
            active_epochs,
            physical_size,
            logical_size,
            reusable_free,
            pressure,
            cumulative_epochs_written,
            cumulative_epochs_deleted,
        }: ChurnSample,
    ) {
        assert_eq!(
            active_epochs,
            cumulative_epochs_written - cumulative_epochs_deleted
        );
        assert_eq!(sequence + 1, cumulative_epochs_written);
        self.observe_state(physical_size, logical_size, reusable_free, pressure);
    }

    fn observe_state(
        &mut self,
        physical_size: u64,
        logical_size: u64,
        reusable_free: u64,
        pressure: u64,
    ) {
        self.peak_physical_size = self.peak_physical_size.max(physical_size);
        self.peak_logical_size = self.peak_logical_size.max(logical_size);
        self.peak_pressure = self.peak_pressure.max(pressure);
        self.peak_reusable_free = self.peak_reusable_free.max(reusable_free);
    }
}

fn write_epoch(file: &File, spec: EpochSpec) {
    let name = spec.name();
    file.root()
        .create_group_with(&name, |group| {
            group.set_attr(
                "sequence",
                AttrValue::I64(i64::try_from(spec.sequence).unwrap()),
            );
            group.set_attr(
                "label",
                AttrValue::String(format!("epoch-{:06}", spec.sequence)),
            );
            group.set_attr("fraction", AttrValue::F64(spec.sequence as f64 + 0.25));
            group.set_attr(
                "integers",
                AttrValue::I32Array(
                    (0..12)
                        .map(|i| i32::try_from(spec.sequence * 100 + i).unwrap())
                        .collect(),
                ),
            );
            for index in 0..MEDIUM_ATTRS {
                group.set_attr(
                    &format!("payload{index}"),
                    AttrValue::U8Array(spec.medium_attr(index)),
                );
            }
            if spec.flavor() == EpochFlavor::DenseMetadata {
                for index in 0..DENSE_EXTRA_ATTRS {
                    group.set_attr(
                        &format!("dense_extra{index}"),
                        AttrValue::I64(i64::try_from(spec.sequence * 10 + index).unwrap()),
                    );
                }
                group.create_dataset(DENSE_ATTR_DATASET, |builder| {
                    builder.with_i32_data(&spec.dense_attr_data());
                    for index in 0..DENSE_DATASET_ATTRS {
                        builder.set_attr(
                            &format!("dense_attr{index}"),
                            AttrValue::I64(i64::try_from(spec.sequence * 100 + index).unwrap()),
                        );
                    }
                });
                group.create_group_with("fanout", |fanout| {
                    for index in 0..FANOUT_CHILDREN {
                        fanout.create_dataset(&format!("child{index:02}"), |builder| {
                            builder.with_i32_data(&[
                                i32::try_from(spec.sequence).unwrap(),
                                i32::try_from(index).unwrap(),
                            ]);
                        });
                    }
                    fanout.create_group_with("nested", |nested| {
                        nested.create_dataset("leaf", |builder| {
                            builder.with_i32_data(&[i32::try_from(spec.sequence).unwrap(), 0x51A7]);
                        });
                    });
                });
            }
        })
        .unwrap();

    create_i32_dataset(
        file,
        &spec.dataset_path(CONTIGUOUS),
        &spec.data(0),
        None,
        &[],
        true,
    );
    create_i32_dataset(
        file,
        &spec.dataset_path(CHUNKED),
        &spec.data(1),
        Some(CHUNK),
        &[],
        false,
    );
    create_i32_dataset(
        file,
        &spec.dataset_path(DEFLATE),
        &spec.compressible_data(),
        Some(CHUNK),
        &[Filter::Deflate(DeflateLevel::Six)],
        false,
    );
    create_i32_dataset(
        file,
        &spec.dataset_path(SHUFFLE_DEFLATE),
        &spec.data(3),
        Some(CHUNK),
        &[Filter::Shuffle, Filter::Deflate(DeflateLevel::Six)],
        false,
    );
    create_i32_dataset(
        file,
        &spec.dataset_path(SCALE_OFFSET),
        &spec.scale_offset_data(),
        Some(CHUNK),
        &[Filter::ScaleOffset(ScaleOffset::Integer(0))],
        false,
    );

    match spec.flavor() {
        EpochFlavor::FixedArray | EpochFlavor::DenseMetadata => {}
        EpochFlavor::Extensible => {
            let data = spec.extensible_data();
            file.root()
                .create_dataset(&spec.dataset_path(EXTENSIBLE), |builder| {
                    builder
                        .with_i32_data(&data)
                        .with_shape(&[u64::try_from(data.len()).unwrap()])
                        .with_maxshape(&[MaxExtent::Unlimited])
                        .with_chunks(&[u64::try_from(CHUNK).unwrap()]);
                })
                .unwrap();
        }
        EpochFlavor::Rank2 => {
            let rank2 = spec.rank2_data();
            file.root()
                .create_dataset(&spec.dataset_path(RANK2), |builder| {
                    builder
                        .with_i32_data(&rank2)
                        .with_shape(&[
                            u64::try_from(RANK2_ROWS).unwrap(),
                            u64::try_from(RANK2_COLS).unwrap(),
                        ])
                        .with_chunks(&[RANK2_CHUNK_ROWS, RANK2_CHUNK_COLS]);
                    dataset::apply_filters(
                        builder,
                        &[Filter::ScaleOffset(ScaleOffset::Integer(0))],
                    );
                })
                .unwrap();
            let single = spec.single_chunk_data();
            file.root()
                .create_dataset(&spec.dataset_path(SINGLE_CHUNK), |builder| {
                    builder
                        .with_i32_data(&single)
                        .with_chunks(&[u64::try_from(SINGLE_CHUNK_LEN).unwrap()]);
                })
                .unwrap();
        }
    }
}

fn create_i32_dataset(
    file: &File,
    path: &str,
    values: &[i32],
    chunk: Option<usize>,
    filters: &[Filter],
    compact_attrs: bool,
) {
    file.root()
        .create_dataset(path, |builder| {
            builder.with_i32_data(values);
            if let Some(chunk) = chunk {
                builder.with_chunks(&[u64::try_from(chunk).unwrap()]);
            }
            dataset::apply_filters(builder, filters);
            if compact_attrs {
                builder
                    .set_attr("attr_layout", AttrValue::String("compact".into()))
                    .set_attr("attr_version", AttrValue::I64(1));
            }
        })
        .unwrap();
}

fn collect_sample(
    path: &Path,
    base: Base,
    file: &File,
    sequence: usize,
    active_epochs: usize,
    cumulative_epochs_written: usize,
    cumulative_epochs_deleted: usize,
) -> ChurnSample {
    let acct = file.space_accounting().unwrap();
    let pressure = pressure(base, &acct);
    let physical_size = base.image_size(path);
    assert!(
        physical_size <= MAX_FILE_BYTES,
        "physical file length {physical_size} exceeded {MAX_FILE_BYTES}"
    );
    ChurnSample {
        sequence,
        active_epochs,
        physical_size,
        logical_size: base.logical_size(&acct),
        reusable_free: acct.reusable_free_bytes,
        pressure,
        cumulative_epochs_written,
        cumulative_epochs_deleted,
    }
}

fn reopen_session(
    path: &Path,
    base: Base,
    session: &mut Option<File>,
    reason: ReopenReason,
    after_epoch: usize,
    live: &VecDeque<usize>,
    epochs_written: usize,
    checkpoints: &mut Vec<ReopenCheckpoint>,
    writable_opens: &mut usize,
    closes: &mut usize,
    reopens: &mut usize,
    stats: &mut RunStats,
    observer: &mut Option<impl FnMut(&Path, &str)>,
) {
    session
        .take()
        .expect("an editing session is open")
        .close()
        .unwrap();
    *closes += 1;
    let physical_size = base.image_size(path);
    assert!(physical_size <= MAX_FILE_BYTES);
    if let Some(observer) = observer {
        observer(
            path,
            &format!("{}-after-epoch-{after_epoch}", reason.label()),
        );
    }
    let persisted_free = persisted_free_bytes(path);
    let reopened = base.open(path);
    let acct = reopened.space_accounting().unwrap();
    let reopened_pressure = pressure(base, &acct);
    assert_eq!(
        acct.reusable_free_bytes, persisted_free,
        "a reopened bounded session must recover all persisted reusable free space"
    );
    validate_reopen_live_data(&reopened, live, epochs_written);
    let ownership = base.exact_ownership(&reopened);
    let recovered_free = acct.reusable_free_bytes;
    stats.observe_state(
        physical_size,
        base.logical_size(&acct),
        recovered_free,
        reopened_pressure,
    );
    *writable_opens += 1;
    *reopens += 1;
    checkpoints.push(ReopenCheckpoint {
        index: checkpoints.len() + 1,
        reason,
        after_epoch,
        physical_size,
        persisted_free,
        recovered_free,
        ownership,
    });
    *session = Some(reopened);
}

fn persisted_free_bytes(path: &Path) -> u64 {
    File::open(path)
        .unwrap()
        .persisted_free_space()
        .unwrap()
        .iter()
        .map(|&(_, len)| len)
        .sum()
}

fn validate_all_epochs(file: &File, live: &VecDeque<usize>, epochs_written: usize) -> Coverage {
    let live_set: BTreeSet<usize> = live.iter().copied().collect();
    let mut coverage = Coverage::default();
    for &sequence in live {
        coverage.add(validate_epoch(file, EpochSpec::new(sequence)));
    }
    for sequence in 0..epochs_written {
        if !live_set.contains(&sequence) {
            assert_deleted_epoch(file, sequence);
        }
    }
    coverage.assert_required();
    coverage
}

fn validate_reopen_live_data(file: &File, live: &VecDeque<usize>, epochs_written: usize) {
    assert_eq!(
        file.dataset("sentinel").unwrap().read_i32().unwrap(),
        vec![7, 11, 13, 17]
    );
    if let Some(&oldest) = live.front() {
        validate_epoch(file, EpochSpec::new(oldest));
    }
    if let Some(&newest) = live.back()
        && live.front().copied() != Some(newest)
    {
        validate_epoch(file, EpochSpec::new(newest));
    }
    if epochs_written > live.len() {
        let first_deleted = (0..epochs_written)
            .find(|sequence| !live.contains(sequence))
            .expect("some epoch has been deleted");
        assert_deleted_epoch(file, first_deleted);
    }
}

fn assert_deleted_epoch(file: &File, sequence: usize) {
    let path = EpochSpec::new(sequence).name();
    let err = file.group(&path).unwrap_err();
    let Error::Format(FormatError::PathNotFound(missing)) = &err else {
        panic!("expected PathNotFound, got {err:?}");
    };
    assert_eq!(missing, &path);
}

fn final_state(path: &Path, base: Base, file: &File, live: &VecDeque<usize>) -> FinalState {
    let acct = file.space_accounting().unwrap();
    let pressure = pressure(base, &acct);
    let physical_size = base.image_size(path);
    assert!(physical_size <= MAX_FILE_BYTES);

    FinalState {
        physical_size,
        logical_size: base.logical_size(&acct),
        reusable_free: acct.reusable_free_bytes,
        reusable_free_regions: acct.reusable_free_space.len(),
        pressure,
        live_epochs: live.len(),
        ownership: base.exact_ownership(file),
    }
}

fn pressure(base: Base, acct: &SpaceAccounting) -> u64 {
    file_space::assert_space_accounting_consistent(acct);
    base.logical_size(acct)
        .checked_sub(acct.reusable_free_bytes)
        .expect("reusable free cannot exceed logical size")
}

fn validate_epoch(file: &File, spec: EpochSpec) -> Coverage {
    let mut coverage = Coverage::default();
    let group = file.group(&spec.name()).unwrap();
    let attrs = group.attrs().unwrap();
    assert_eq!(
        attrs.get("sequence"),
        Some(&AttrValue::I64(i64::try_from(spec.sequence).unwrap()))
    );
    assert_eq!(
        attrs.get("label"),
        Some(&AttrValue::String(format!("epoch-{:06}", spec.sequence)))
    );
    assert_eq!(
        attrs.get("fraction"),
        Some(&AttrValue::F64(spec.sequence as f64 + 0.25))
    );
    assert_eq!(
        attrs.get("integers"),
        Some(&AttrValue::I32Array(
            (0..12)
                .map(|i| i32::try_from(spec.sequence * 100 + i).unwrap())
                .collect()
        ))
    );
    for index in 0..MEDIUM_ATTRS {
        assert_eq!(
            attrs.get(&format!("payload{index}")),
            Some(&AttrValue::U8Array(spec.medium_attr(index)))
        );
    }
    coverage.compact_link_groups += 1;
    if spec.flavor() == EpochFlavor::DenseMetadata {
        assert_eq!(attrs.len(), 8 + DENSE_EXTRA_ATTRS);
        for index in 0..DENSE_EXTRA_ATTRS {
            assert_eq!(
                attrs.get(&format!("dense_extra{index}")),
                Some(&AttrValue::I64(
                    i64::try_from(spec.sequence * 10 + index).unwrap()
                ))
            );
        }
        coverage.dense_attr_objects += 1;
    } else {
        assert_eq!(attrs.len(), 8);
        coverage.compact_attr_objects += 1;
    }

    coverage.add(validate_dataset(
        file,
        &spec.dataset_path(CONTIGUOUS),
        spec.data(0),
        &[u64::try_from(spec.len).unwrap()],
        None,
        None,
        &[],
    ));
    let contiguous_attrs = file
        .dataset(&spec.dataset_path(CONTIGUOUS))
        .unwrap()
        .attrs()
        .unwrap();
    assert_eq!(
        contiguous_attrs.get("attr_layout"),
        Some(&AttrValue::String("compact".into()))
    );
    assert_eq!(
        contiguous_attrs.get("attr_version"),
        Some(&AttrValue::I64(1))
    );
    assert_eq!(contiguous_attrs.len(), 2);
    coverage.compact_attr_objects += 1;

    coverage.add(validate_dataset(
        file,
        &spec.dataset_path(CHUNKED),
        spec.data(1),
        &[u64::try_from(spec.len).unwrap()],
        Some(&[u64::try_from(CHUNK).unwrap()]),
        Some(ChunkIndex::FixedArray),
        &[],
    ));
    coverage.add(validate_dataset(
        file,
        &spec.dataset_path(DEFLATE),
        spec.compressible_data(),
        &[u64::try_from(spec.len).unwrap()],
        Some(&[u64::try_from(CHUNK).unwrap()]),
        Some(ChunkIndex::FixedArray),
        &[FILTER_DEFLATE],
    ));
    let deflate_stored: u64 = file
        .dataset(&spec.dataset_path(DEFLATE))
        .unwrap()
        .chunks()
        .unwrap()
        .iter()
        .map(|chunk| chunk.storage_size)
        .sum();
    assert!(
        deflate_stored < u64::try_from(spec.len * std::mem::size_of::<i32>()).unwrap() / 2,
        "the compressible deflate lane must actually compress"
    );
    coverage.compressible_filtered += 1;

    coverage.add(validate_dataset(
        file,
        &spec.dataset_path(SHUFFLE_DEFLATE),
        spec.data(3),
        &[u64::try_from(spec.len).unwrap()],
        Some(&[u64::try_from(CHUNK).unwrap()]),
        Some(ChunkIndex::FixedArray),
        &[FILTER_SHUFFLE, FILTER_DEFLATE],
    ));
    let shuffled_stored: u64 = file
        .dataset(&spec.dataset_path(SHUFFLE_DEFLATE))
        .unwrap()
        .chunks()
        .unwrap()
        .iter()
        .map(|chunk| chunk.storage_size)
        .sum();
    assert!(
        shuffled_stored > u64::try_from(spec.len * std::mem::size_of::<i32>()).unwrap() / 2,
        "the incompressible lane must stay materially raw-like after filtering"
    );
    coverage.incompressible_filtered += 1;

    coverage.add(validate_dataset(
        file,
        &spec.dataset_path(SCALE_OFFSET),
        spec.scale_offset_data(),
        &[u64::try_from(spec.len).unwrap()],
        Some(&[u64::try_from(CHUNK).unwrap()]),
        Some(ChunkIndex::FixedArray),
        &[FILTER_SCALE_OFFSET],
    ));

    match spec.flavor() {
        EpochFlavor::FixedArray => {}
        EpochFlavor::DenseMetadata => {
            let dense_path = spec.dataset_path(DENSE_ATTR_DATASET);
            coverage.add(validate_dataset(
                file,
                &dense_path,
                spec.dense_attr_data(),
                &[u64::try_from(spec.dense_attr_data().len()).unwrap()],
                None,
                None,
                &[],
            ));
            let dense_attrs = file.dataset(&dense_path).unwrap().attrs().unwrap();
            assert_eq!(dense_attrs.len(), DENSE_DATASET_ATTRS);
            for index in 0..DENSE_DATASET_ATTRS {
                assert_eq!(
                    dense_attrs.get(&format!("dense_attr{index}")),
                    Some(&AttrValue::I64(
                        i64::try_from(spec.sequence * 100 + index).unwrap()
                    ))
                );
            }
            coverage.dense_attr_objects += 1;

            let fanout_path = format!("{}/fanout", spec.name());
            let fanout = file.group(&fanout_path).unwrap();
            assert_eq!(fanout.datasets().unwrap().len(), FANOUT_CHILDREN);
            assert_eq!(fanout.groups().unwrap(), vec!["nested".to_string()]);
            for index in 0..FANOUT_CHILDREN {
                assert_eq!(
                    file.dataset(&format!("{fanout_path}/child{index:02}"))
                        .unwrap()
                        .read_i32()
                        .unwrap(),
                    vec![
                        i32::try_from(spec.sequence).unwrap(),
                        i32::try_from(index).unwrap()
                    ]
                );
            }
            assert_eq!(
                file.dataset(&format!("{fanout_path}/nested/leaf"))
                    .unwrap()
                    .read_i32()
                    .unwrap(),
                vec![i32::try_from(spec.sequence).unwrap(), 0x51A7]
            );
            coverage.nested_epoch_groups += 2;
            coverage.compact_link_groups += 2;
            coverage.high_fanout_compact_link_groups += 1;
        }
        EpochFlavor::Extensible => {
            let path = spec.dataset_path(EXTENSIBLE);
            let expected = spec.extensible_data();
            coverage.add(validate_dataset(
                file,
                &path,
                expected.clone(),
                &[u64::try_from(expected.len()).unwrap()],
                Some(&[u64::try_from(CHUNK).unwrap()]),
                Some(ChunkIndex::ExtensibleArray),
                &[],
            ));
            assert_eq!(
                file.dataset(&path).unwrap().maxshape().unwrap(),
                Some(vec![MaxExtent::Unlimited])
            );
        }
        EpochFlavor::Rank2 => {
            coverage.add(validate_dataset(
                file,
                &spec.dataset_path(RANK2),
                spec.rank2_data(),
                &[
                    u64::try_from(RANK2_ROWS).unwrap(),
                    u64::try_from(RANK2_COLS).unwrap(),
                ],
                Some(&[RANK2_CHUNK_ROWS, RANK2_CHUNK_COLS]),
                Some(ChunkIndex::FixedArray),
                &[FILTER_SCALE_OFFSET],
            ));
            coverage.add(validate_dataset(
                file,
                &spec.dataset_path(SINGLE_CHUNK),
                spec.single_chunk_data(),
                &[u64::try_from(SINGLE_CHUNK_LEN).unwrap()],
                Some(&[u64::try_from(SINGLE_CHUNK_LEN).unwrap()]),
                Some(ChunkIndex::SingleChunk),
                &[],
            ));
        }
    }
    coverage
}

fn validate_dataset(
    file: &File,
    path: &str,
    expected: Vec<i32>,
    expected_shape: &[u64],
    expected_chunk_shape: Option<&[u64]>,
    expected_index: Option<ChunkIndex>,
    expected_filters: &[u16],
) -> Coverage {
    let mut coverage = Coverage::default();
    let dataset = file.dataset(path).unwrap();
    let shape = dataset.shape().unwrap();
    assert_eq!(shape, expected_shape);
    match shape.len() {
        1 => coverage.rank1_datasets += 1,
        2 => coverage.rank2_datasets += 1,
        rank => panic!("{path}: expected rank 1 or 2, got {rank}"),
    }
    match (
        expected_chunk_shape,
        expected_index,
        dataset.layout().unwrap(),
    ) {
        (None, None, Layout::Contiguous { .. }) => {}
        (
            Some(expected_chunk_shape),
            Some(expected_index),
            Layout::Chunked { chunk_shape, index },
        ) => {
            assert_eq!(chunk_shape, expected_chunk_shape);
            assert_eq!(index, expected_index);
            assert_eq!(dataset.chunk_index().unwrap(), Some(expected_index));
            match index {
                ChunkIndex::SingleChunk => coverage.single_chunk += 1,
                ChunkIndex::FixedArray => coverage.fixed_array += 1,
                ChunkIndex::ExtensibleArray => coverage.extensible_array += 1,
                other => panic!("{path}: unexpected chunk index {other:?}"),
            }
            if shape
                .iter()
                .zip(&chunk_shape)
                .all(|(&dimension, &chunk)| dimension % chunk == 0)
            {
                coverage.aligned_chunk_grids += 1;
            } else {
                coverage.partial_edge_chunk_grids += 1;
            }
        }
        (_, _, layout) => panic!("{path}: unexpected layout {layout:?}"),
    }
    assert_eq!(
        dataset.filters().unwrap(),
        expected_filters
            .iter()
            .map(|&id| FilterId::try_from(id).unwrap())
            .collect::<Vec<_>>()
    );
    match expected_filters {
        [] => coverage.filter_none += 1,
        [FILTER_DEFLATE] => coverage.filter_deflate += 1,
        [FILTER_SHUFFLE, FILTER_DEFLATE] => coverage.filter_shuffle_deflate += 1,
        [FILTER_SCALE_OFFSET] => coverage.filter_scale_offset += 1,
        other => panic!("{path}: unexpected expected filter pipeline {other:?}"),
    }
    assert_eq!(dataset.read_i32().unwrap(), expected);
    coverage
}

#[derive(Clone, Copy)]
struct EpochSpec {
    sequence: usize,
    len: usize,
}

impl EpochSpec {
    fn new(sequence: usize) -> Self {
        let len = if sequence % 2 == 0 {
            CHUNK * WHOLE_CHUNKS
        } else {
            CHUNK * WHOLE_CHUNKS + 137
        };
        Self { sequence, len }
    }

    fn dataset_path(self, dataset: &str) -> String {
        format!("{}/{dataset}", self.name())
    }

    fn name(self) -> String {
        format!("epoch{:06}", self.sequence)
    }

    fn flavor(self) -> EpochFlavor {
        EpochFlavor::for_sequence(self.sequence)
    }

    fn data(self, lane: u32) -> Vec<i32> {
        dataset::incompressible(self.seed(lane), self.len)
    }

    fn compressible_data(self) -> Vec<i32> {
        let Self { sequence, len } = self;
        let sequence = i32::try_from(sequence).expect("the churn sequence fits in i32");
        (0..len)
            .map(|i| i32::try_from(i / 256).unwrap().wrapping_add(sequence) & 0x0f)
            .collect()
    }

    fn extensible_data(self) -> Vec<i32> {
        let len = CHUNK * EXTENSIBLE_CHUNKS + self.sequence % 137;
        dataset::incompressible(self.seed(5), len)
    }

    fn rank2_data(self) -> Vec<i32> {
        let sequence = i32::try_from(self.sequence).expect("the churn sequence fits in i32");
        (0..RANK2_ROWS * RANK2_COLS)
            .map(|i| {
                let i = i32::try_from(i).expect("the rank-2 fixture length fits in i32");
                ((i * 11 + sequence * 37) % 8192) - 4096
            })
            .collect()
    }

    fn single_chunk_data(self) -> Vec<i32> {
        dataset::incompressible(self.seed(6), SINGLE_CHUNK_LEN)
    }

    fn dense_attr_data(self) -> Vec<i32> {
        dataset::incompressible(self.seed(7), CHUNK * 4 + self.sequence % 31)
    }

    fn scale_offset_data(self) -> Vec<i32> {
        let Self { sequence, len } = self;
        let sequence = i32::try_from(sequence).expect("the churn sequence fits in i32");
        (0..len)
            .map(|i| {
                let i = i32::try_from(i).expect("the epoch length fits in i32");
                ((i + sequence * 37) % 4096) - 2048
            })
            .collect()
    }

    fn medium_attr(self, index: usize) -> Vec<u8> {
        let sequence = u8::try_from(self.sequence % 251).unwrap();
        let index = u8::try_from(index).unwrap();
        (0..MEDIUM_ATTR_LEN)
            .map(|i| {
                let i = u8::try_from(i % 251).unwrap();
                sequence
                    .wrapping_mul(17)
                    .wrapping_add(index.wrapping_mul(43))
                    .wrapping_add(i)
            })
            .collect()
    }

    fn seed(self, lane: u32) -> u32 {
        let sequence = u32::try_from(self.sequence).expect("the churn sequence fits in u32");
        0xA511_E9B3u32
            .wrapping_add(sequence.wrapping_mul(0x9E37_79B9))
            .wrapping_add(lane.wrapping_mul(0x85EB_CA6B))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EpochFlavor {
    DenseMetadata,
    Extensible,
    FixedArray,
    Rank2,
}

impl EpochFlavor {
    fn for_sequence(sequence: usize) -> Self {
        match sequence % 4 {
            0 => Self::FixedArray,
            1 => Self::DenseMetadata,
            2 => Self::Extensible,
            _ => Self::Rank2,
        }
    }
}

/// The userblock configuration for a churn file.
///
/// Reports subtract the userblock length from physical and logical file sizes. Files with a
/// userblock use mirrored editing. Files with a zero base use bounded editing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Base {
    /// Reserves a page for a stamped userblock.
    Userblock,
    /// Starts the HDF5 superblock at the beginning of the file.
    Zero,
}

impl Base {
    const fn label(self) -> &'static str {
        match self {
            Self::Userblock => "userblock",
            Self::Zero => "zero",
        }
    }

    const fn userblock(self) -> u64 {
        match self {
            Self::Userblock => USERBLOCK_BYTES,
            Self::Zero => 0,
        }
    }

    fn write(self, path: &Path, mut builder: FileBuilder) -> Option<userblock::Userblock> {
        match self {
            Self::Userblock => {
                builder.with_userblock(UserblockSize::try_from(USERBLOCK_BYTES).unwrap());
                Some(userblock::write(
                    path,
                    builder,
                    usize::try_from(USERBLOCK_BYTES).unwrap(),
                    b"churn",
                ))
            }
            Self::Zero => {
                builder.write(path).unwrap();
                None
            }
        }
    }

    fn open(self, path: &Path) -> File {
        match self {
            Self::Userblock => File::open_rw_with_options(
                path,
                FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Mirrored),
            )
            .unwrap(),
            Self::Zero => session::open_bounded(path).unwrap(),
        }
    }

    fn image_size(self, path: &Path) -> u64 {
        std::fs::metadata(path).unwrap().len() - self.userblock()
    }

    fn logical_size(self, acct: &SpaceAccounting) -> u64 {
        acct.logical_size - self.userblock()
    }

    #[track_caller]
    fn exact_ownership(self, file: &File) -> SpaceOwnershipSummary {
        let ownership = file_space::assert_exact_space_ownership(file);
        assert_eq!(ownership.userblock_bytes, self.userblock());
        SpaceOwnershipSummary {
            physical_size: ownership.physical_size - self.userblock(),
            logical_size: ownership.logical_size - self.userblock(),
            ..ownership
        }
    }
}

/// The persistent allocation strategy used by the churn workload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Strategy {
    /// Uses free-space managers with aggregation.
    FsmAggr,
    /// Uses paged aggregation with persistent free-space managers.
    Page,
}

impl Strategy {
    const fn label(self) -> &'static str {
        match self {
            Self::FsmAggr => "fsm-aggr",
            Self::Page => "page",
        }
    }

    /// Returns `name` for free-space aggregation, or appends `_page` for paged aggregation.
    pub fn snapshot(self, name: &str) -> String {
        match self {
            Self::FsmAggr => name.to_owned(),
            Self::Page => format!("{name}_page"),
        }
    }

    fn configure(self, builder: &mut FileBuilder) {
        match self {
            Self::FsmAggr => {
                builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
            }
            Self::Page => {
                builder
                    .with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
                    .with_file_space_page_size(FileSpacePageSize::try_from(PAGE_BYTES).unwrap());
            }
        }
    }
}

/// The points at which the churn workload closes and reopens its editing session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifecycle {
    /// Keeps one editing session open through the final data checks.
    ContinuousOpen,
    /// Reopens once before the final data checks.
    ReopenBeforeFinalCheck,
    /// Reopens during initial filling and after each deletion wave.
    ReopenFrequently,
}

impl Lifecycle {
    const fn label(self) -> &'static str {
        match self {
            Self::ContinuousOpen => "continuous-open",
            Self::ReopenBeforeFinalCheck => "reopen-before-final-check",
            Self::ReopenFrequently => "reopen-frequently",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReopenReason {
    DeletionWave,
    FinalCheck,
    InitialFill,
}

impl ReopenReason {
    const fn label(self) -> &'static str {
        match self {
            Self::InitialFill => "initial-fill",
            Self::DeletionWave => "deletion-wave",
            Self::FinalCheck => "final-check",
        }
    }
}

#[derive(Clone, Copy)]
struct ChurnSample {
    sequence: usize,
    active_epochs: usize,
    physical_size: u64,
    logical_size: u64,
    reusable_free: u64,
    pressure: u64,
    cumulative_epochs_written: usize,
    cumulative_epochs_deleted: usize,
}

struct DeletionWave {
    index: usize,
    after_epoch: usize,
    active_before: usize,
    deleted: usize,
    active_after: usize,
    physical_size: u64,
    logical_size: u64,
    reusable_free: u64,
    pressure: u64,
    ownership: SpaceOwnershipSummary,
}

struct ReopenCheckpoint {
    index: usize,
    reason: ReopenReason,
    after_epoch: usize,
    physical_size: u64,
    persisted_free: u64,
    recovered_free: u64,
    ownership: SpaceOwnershipSummary,
}

struct FinalState {
    physical_size: u64,
    logical_size: u64,
    reusable_free: u64,
    reusable_free_regions: usize,
    pressure: u64,
    live_epochs: usize,
    ownership: SpaceOwnershipSummary,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Coverage {
    nested_epoch_groups: usize,
    compact_attr_objects: usize,
    dense_attr_objects: usize,
    compact_link_groups: usize,
    high_fanout_compact_link_groups: usize,
    dense_link_groups: usize,
    single_chunk: usize,
    fixed_array: usize,
    extensible_array: usize,
    rank1_datasets: usize,
    rank2_datasets: usize,
    aligned_chunk_grids: usize,
    partial_edge_chunk_grids: usize,
    filter_none: usize,
    filter_deflate: usize,
    filter_shuffle_deflate: usize,
    filter_scale_offset: usize,
    compressible_filtered: usize,
    incompressible_filtered: usize,
}

impl Coverage {
    fn add(
        &mut self,
        Coverage {
            nested_epoch_groups,
            compact_attr_objects,
            dense_attr_objects,
            compact_link_groups,
            high_fanout_compact_link_groups,
            dense_link_groups,
            single_chunk,
            fixed_array,
            extensible_array,
            rank1_datasets,
            rank2_datasets,
            aligned_chunk_grids,
            partial_edge_chunk_grids,
            filter_none,
            filter_deflate,
            filter_shuffle_deflate,
            filter_scale_offset,
            compressible_filtered,
            incompressible_filtered,
        }: Self,
    ) {
        self.nested_epoch_groups += nested_epoch_groups;
        self.compact_attr_objects += compact_attr_objects;
        self.dense_attr_objects += dense_attr_objects;
        self.compact_link_groups += compact_link_groups;
        self.high_fanout_compact_link_groups += high_fanout_compact_link_groups;
        self.dense_link_groups += dense_link_groups;
        self.single_chunk += single_chunk;
        self.fixed_array += fixed_array;
        self.extensible_array += extensible_array;
        self.rank1_datasets += rank1_datasets;
        self.rank2_datasets += rank2_datasets;
        self.aligned_chunk_grids += aligned_chunk_grids;
        self.partial_edge_chunk_grids += partial_edge_chunk_grids;
        self.filter_none += filter_none;
        self.filter_deflate += filter_deflate;
        self.filter_shuffle_deflate += filter_shuffle_deflate;
        self.filter_scale_offset += filter_scale_offset;
        self.compressible_filtered += compressible_filtered;
        self.incompressible_filtered += incompressible_filtered;
    }

    fn assert_required(&self) {
        let Coverage {
            nested_epoch_groups,
            compact_attr_objects,
            dense_attr_objects,
            compact_link_groups,
            high_fanout_compact_link_groups,
            dense_link_groups,
            single_chunk,
            fixed_array,
            extensible_array,
            rank1_datasets,
            rank2_datasets,
            aligned_chunk_grids,
            partial_edge_chunk_grids,
            filter_none,
            filter_deflate,
            filter_shuffle_deflate,
            filter_scale_offset,
            compressible_filtered,
            incompressible_filtered,
        } = self;
        assert!(*nested_epoch_groups > 0);
        assert!(*compact_attr_objects > 0);
        assert!(*dense_attr_objects > 0);
        assert!(*compact_link_groups > 0);
        assert!(*high_fanout_compact_link_groups > 0);
        assert_eq!(
            *dense_link_groups, 0,
            "hdf5-pure's writer intentionally keeps group links compact (issue #131)"
        );
        assert!(*single_chunk > 0);
        assert!(*fixed_array > 0);
        assert!(*extensible_array > 0);
        assert!(*rank1_datasets > 0);
        assert!(*rank2_datasets > 0);
        assert!(*aligned_chunk_grids > 0);
        assert!(*partial_edge_chunk_grids > 0);
        assert!(*filter_none > 0);
        assert!(*filter_deflate > 0);
        assert!(*filter_shuffle_deflate > 0);
        assert!(*filter_scale_offset > 0);
        assert!(*compressible_filtered > 0);
        assert!(*incompressible_filtered > 0);
    }
}

const MAX_FILE_BYTES: u64 = 32 * 1024 * 1024;
const USERBLOCK_BYTES: u64 = PAGE_BYTES;
const PAGE_BYTES: u64 = 4096;
const HEADROOM_BYTES: u64 = MAX_FILE_BYTES / 8;
const HIGH_WATER_BYTES: u64 = MAX_FILE_BYTES - HEADROOM_BYTES;
const LOW_WATER_BYTES: u64 = 24 * 1024 * 1024;
const CHURN_EPOCHS_AFTER_FIRST_DELETE: usize = 64;
const CHUNK: usize = 1024;
const WHOLE_CHUNKS: usize = 80;
const INITIAL_REOPEN_INTERVAL: usize = 8;
const MEDIUM_ATTRS: usize = 4;
const MEDIUM_ATTR_LEN: usize = 256;
const DENSE_EXTRA_ATTRS: usize = 4;
const DENSE_DATASET_ATTRS: usize = 10;
const FANOUT_CHILDREN: usize = 12;
const EXTENSIBLE_CHUNKS: usize = 20;
const RANK2_ROWS: usize = 257;
const RANK2_COLS: usize = 263;
const RANK2_CHUNK_ROWS: u64 = 64;
const RANK2_CHUNK_COLS: u64 = 65;
const SINGLE_CHUNK_LEN: usize = 257;
const CONTIGUOUS: &str = "contiguous";
const CHUNKED: &str = "chunked";
const DEFLATE: &str = "deflate";
const SHUFFLE_DEFLATE: &str = "shuffle_deflate";
const SCALE_OFFSET: &str = "scale_offset";
const DENSE_ATTR_DATASET: &str = "dense_attr_dataset";
const EXTENSIBLE: &str = "extensible";
const RANK2: &str = "rank2";
const SINGLE_CHUNK: &str = "single_chunk";
/// The deflate filter identifier defined in "The Data Storage - Filter Pipeline Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
const FILTER_DEFLATE: u16 = 1;
/// The shuffle filter identifier defined in "The Data Storage - Filter Pipeline Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
const FILTER_SHUFFLE: u16 = 2;
/// The scale-offset filter identifier defined in "The Data Storage - Filter Pipeline Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
const FILTER_SCALE_OFFSET: u16 = 6;
