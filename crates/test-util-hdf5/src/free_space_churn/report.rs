//! Text reports for the churn workload.

use test_util::snapshot::text::{Report, Value};

use super::{
    ChurnReport, Coverage, DeletionWave, FinalState, HEADROOM_BYTES, HIGH_WATER_BYTES,
    LOW_WATER_BYTES, MAX_FILE_BYTES, ReopenCheckpoint,
};

impl ChurnReport {
    /// Formats workload limits, deletion waves, reopen checkpoints, and free-space totals.
    ///
    /// Numbers are grouped by thousands, and numeric columns are right-aligned.
    pub fn render(&self) -> String {
        let Self {
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
            after_close_free,
            after_close_physical_size,
            after_close_logical_size,
        } = self;
        let mut report = Report::new();
        report.table(
            "free-space churn",
            ["property", "value"],
            [[Value::text("scenario"), Value::text(lifecycle.label())]],
        );
        report.metrics(
            "limits",
            "bytes",
            [
                ("maximum file size", MAX_FILE_BYTES),
                ("high-water", HIGH_WATER_BYTES),
                ("low-water", LOW_WATER_BYTES),
                ("headroom", HEADROOM_BYTES),
            ],
        );
        report.metrics(
            "workload",
            "count",
            [
                ("epochs written", *epochs_written as u64),
                ("epochs deleted", *epochs_deleted as u64),
                ("epochs still live", *epochs_live as u64),
                (
                    "epochs before first delete",
                    *epochs_before_first_delete as u64,
                ),
                ("churn epochs", *churn_epochs_after_first_delete as u64),
                ("deletion waves", deletion_waves.len() as u64),
                (
                    "max deleted in one wave",
                    *max_epochs_deleted_in_one_wave as u64,
                ),
            ],
        );
        report.metrics(
            "lifecycle",
            "count",
            [
                ("writable opens", *writable_opens as u64),
                ("closes", *closes as u64),
                ("reopens", *reopens as u64),
            ],
        );
        report.metrics(
            "high water",
            "bytes",
            [
                ("physical file", *peak_physical_size),
                ("logical file", *peak_logical_size),
                ("pressure", *peak_pressure),
                ("reusable free", *peak_reusable_free),
            ],
        );
        report.metrics(
            "reuse",
            "count",
            [
                (
                    "epoch writes growing physical file",
                    *epoch_writes_growing_physical_file as u64,
                ),
                (
                    "epoch writes without physical growth",
                    *epoch_writes_without_physical_growth as u64,
                ),
                (
                    "post-churn writes without growth",
                    *post_churn_epoch_writes_without_physical_growth as u64,
                ),
            ],
        );
        report.table(
            "deletion waves",
            [
                "wave",
                "after-epoch",
                "live-before",
                "deleted",
                "live-after",
                "physical",
                "logical",
                "reusable",
                "pressure",
            ],
            deletion_waves.iter().map(DeletionWave::values),
        );
        report.table(
            "reopen checkpoints",
            [
                "#",
                "reason",
                "after-epoch",
                "physical",
                "persisted-free",
                "recovered-free",
            ],
            reopen_checkpoints.iter().map(ReopenCheckpoint::values),
        );
        final_state.append_report(&mut report);
        coverage.append_report(&mut report);
        report.metrics(
            "persisted space after writer close",
            "value",
            [
                ("physical file size (bytes)", *after_close_physical_size),
                ("logical EOF (bytes)", *after_close_logical_size),
                ("free bytes", after_close_free.free_sections().free_bytes()),
                (
                    "sections (count)",
                    after_close_free.free_sections().section_count() as u64,
                ),
            ],
        );
        report.finish()
    }
}

impl DeletionWave {
    fn values(&self) -> [Value; 9] {
        let Self {
            index,
            after_epoch,
            active_before,
            deleted,
            active_after,
            physical_size,
            logical_size,
            reusable_free,
            pressure,
        } = self;
        [
            Value::Number(*index as u64),
            Value::Number(*after_epoch as u64),
            Value::Number(*active_before as u64),
            Value::Number(*deleted as u64),
            Value::Number(*active_after as u64),
            Value::Number(*physical_size),
            Value::Number(*logical_size),
            Value::Number(*reusable_free),
            Value::Number(*pressure),
        ]
    }
}

impl ReopenCheckpoint {
    fn values(&self) -> [Value; 6] {
        let Self {
            index,
            reason,
            after_epoch,
            physical_size,
            persisted_free,
            recovered_free,
        } = self;
        [
            Value::Number(*index as u64),
            Value::text(reason.label()),
            Value::Number(*after_epoch as u64),
            Value::Number(*physical_size),
            Value::Number(*persisted_free),
            Value::Number(*recovered_free),
        ]
    }
}

impl FinalState {
    fn append_report(&self, report: &mut Report) {
        let Self {
            physical_size,
            logical_size,
            reusable_free,
            reusable_free_regions,
            pressure,
            live_epochs,
        } = self;
        report.metrics(
            "final",
            "value",
            [
                ("physical file size (bytes)", *physical_size),
                ("logical size (bytes)", *logical_size),
                ("reusable free (bytes)", *reusable_free),
                (
                    "reusable free regions (count)",
                    *reusable_free_regions as u64,
                ),
                ("pressure (bytes)", *pressure),
                ("live epochs (count)", *live_epochs as u64),
            ],
        );
    }
}

impl Coverage {
    fn append_report(&self, report: &mut Report) {
        let Self {
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
        report.metrics(
            "coverage: object/group layout",
            "count",
            [
                ("nested epoch groups", *nested_epoch_groups as u64),
                ("compact attribute objects", *compact_attr_objects as u64),
                ("dense attribute objects", *dense_attr_objects as u64),
                ("compact-link groups", *compact_link_groups as u64),
                (
                    "high-fanout compact-link groups",
                    *high_fanout_compact_link_groups as u64,
                ),
                (
                    "dense-link groups (writer unsupported)",
                    *dense_link_groups as u64,
                ),
            ],
        );
        report.metrics(
            "coverage: chunk indices",
            "count",
            [
                ("single chunk", *single_chunk as u64),
                ("fixed array", *fixed_array as u64),
                ("extensible array", *extensible_array as u64),
            ],
        );
        report.metrics(
            "coverage: ranks",
            "count",
            [
                ("rank 1 datasets", *rank1_datasets as u64),
                ("rank 2 datasets", *rank2_datasets as u64),
            ],
        );
        report.metrics(
            "coverage: chunk grids",
            "count",
            [
                ("aligned", *aligned_chunk_grids as u64),
                ("partial-edge", *partial_edge_chunk_grids as u64),
            ],
        );
        report.metrics(
            "coverage: filters",
            "count",
            [
                ("none", *filter_none as u64),
                ("deflate", *filter_deflate as u64),
                ("shuffle + deflate", *filter_shuffle_deflate as u64),
                ("integer scale-offset", *filter_scale_offset as u64),
            ],
        );
        report.metrics(
            "coverage: filtered contents",
            "count",
            [
                ("compressible", *compressible_filtered as u64),
                ("incompressible", *incompressible_filtered as u64),
            ],
        );
    }
}
