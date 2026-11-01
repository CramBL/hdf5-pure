//! Text reports for the endurance workload.

use hdf5_pure::FileSpaceStrategy;
use test_util::snapshot::text::{Report, Value};

/// Measurements from the final reopened session, with persisted space from the preceding close.
pub(super) struct EnduranceReport {
    pub(super) iteration: usize,
    pub(super) strategy: FileSpaceStrategy,
    pub(super) persist: bool,
    pub(super) threshold: u64,
    pub(super) page_size: Option<u64>,
    pub(super) physical: u64,
    pub(super) logical_size: u64,
    pub(super) groups: usize,
    pub(super) reusable_bytes: u64,
    pub(super) reusable_sections: usize,
    /// The byte and section counts from the preceding close.
    pub(super) persisted: Option<(u64, usize)>,
}

impl EnduranceReport {
    pub(super) fn render(self) -> String {
        let Self {
            iteration,
            strategy,
            persist,
            threshold,
            page_size,
            physical,
            logical_size,
            groups,
            reusable_bytes,
            reusable_sections,
            persisted,
        } = self;
        let mut report = Report::new();
        report.table(
            "small-file free-space endurance",
            ["measurement", "value"],
            [
                [Value::text("state"), Value::text("final reopen")],
                [Value::text("cycles"), Value::Number(iteration as u64)],
            ],
        );
        let mut settings = vec![
            [
                Value::text("strategy"),
                Value::text(format!("{strategy:?}")),
            ],
            [Value::text("persistence"), Value::text(persist)],
            [Value::text("threshold (bytes)"), Value::Number(threshold)],
        ];
        if let Some(page_size) = page_size {
            settings.push([Value::text("page size (bytes)"), Value::Number(page_size)]);
        }
        report.table("creation settings", ["property", "value"], settings);
        report.metrics(
            "file",
            "value",
            [
                ("physical size (bytes)", physical),
                ("logical EOF (bytes)", logical_size),
                ("top-level groups (count)", groups as u64),
            ],
        );
        report.metrics(
            "reusable space (writable session)",
            "value",
            [
                ("bytes", reusable_bytes),
                ("sections (count)", reusable_sections as u64),
            ],
        );
        match persisted {
            Some((bytes, sections)) => report.metrics(
                "persisted space (last close)",
                "value",
                [("bytes", bytes), ("sections (count)", sections as u64)],
            ),
            None => report.table(
                "persisted space (last close)",
                ["status"],
                [[Value::text("persistence not configured")]],
            ),
        }
        report.finish()
    }
}
