use test_util::snapshot::text::{Report, Value};
use test_util_hdf5::aggregation::Object;
use test_util_hdf5::aggregation::Step;

use crate::edit::SpaceOwnership;

/// Writes the operation and an extent table sorted by offset within each category.
///
/// # Panics
///
/// Panics if an extent's end offset exceeds [`u64::MAX`].
pub(super) fn write_page_step(
    report: &mut String,
    number: usize,
    step: Step,
    ownership: SpaceOwnership,
) {
    let mut tables = Report::new();
    let (operation, name, elements) = match step {
        Step::Create(Object::Dataset(name, len)) => (
            "Create dataset",
            name,
            Value::Number(u64::try_from(len).unwrap()),
        ),
        Step::Create(Object::Group(name)) => ("Create group", name, Value::text("-")),
        Step::Create(Object::Strings(name)) => (
            "Create variable-length string dataset",
            name,
            Value::text("-"),
        ),
        Step::Delete(name) => ("Delete link", name, Value::text("-")),
    };
    tables.table(
        "Operation",
        ["Step", "Operation", "Object", "i32 elements"],
        [[
            Value::Number(u64::try_from(number).unwrap()),
            Value::text(operation),
            Value::text(format!("{name:?}")),
            elements,
        ]],
    );
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
    } = ownership;
    tables.table(
        "Allocation summary (bytes)",
        ["Physical size", "Logical size", "Analysis"],
        [[
            Value::Number(physical_size),
            Value::Number(logical_size),
            Value::text(if analysis_complete {
                "complete"
            } else {
                "incomplete"
            }),
        ]],
    );
    let mut empty = Vec::new();
    tables.table(
        "Allocation extents (bytes, exclusive end)",
        ["Category", "Start", "End", "Length"],
        [
            ("Userblock", userblock_space),
            ("Live metadata", live_metadata_space),
            ("Live raw data", live_raw_space),
            ("Free-space managers", free_space_manager_space),
            ("Reusable free space", reusable_free_space),
            ("Unclassified", unclassified_space),
            ("Trailing slack", trailing_slack_space),
        ]
        .into_iter()
        .flat_map(|(category, mut extents)| {
            if extents.is_empty() {
                empty.push([Value::text(category)]);
            }
            extents.sort_unstable();
            extents.into_iter().map(move |(start, length)| {
                [
                    Value::text(category),
                    Value::Number(start),
                    Value::Number(start.checked_add(length).unwrap()),
                    Value::Number(length),
                ]
            })
        }),
    );
    if !empty.is_empty() {
        tables.table("Categories without extents", ["Category"], empty);
    }
    report.push('\n');
    report.push_str(&tables.finish());
    report.push('\n');
}
