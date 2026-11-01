use std::fmt::Write as _;

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
    write!(report, "\nStep {number}: ").unwrap();
    match step {
        Step::Create(Object::Dataset(name, len)) => {
            writeln!(report, "Create dataset {name:?} with {len} i32 elements").unwrap();
        }
        Step::Create(Object::Group(name)) => writeln!(report, "Create group {name:?}").unwrap(),
        Step::Create(Object::Strings(name)) => {
            writeln!(report, "Create variable-length string dataset {name:?}").unwrap();
        }
        Step::Delete(name) => writeln!(report, "Delete link {name:?}").unwrap(),
    }
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
    writeln!(
        report,
        "  Physical size: {physical_size}, logical size: {logical_size}, analysis: {}",
        if analysis_complete {
            "complete"
        } else {
            "incomplete"
        }
    )
    .unwrap();
    writeln!(
        report,
        "  {:<19} {:>10} {:>10} {:>10}",
        "Category", "Start", "End", "Length"
    )
    .unwrap();
    let mut empty = Vec::new();
    for (category, mut extents) in [
        ("Userblock", userblock_space),
        ("Live metadata", live_metadata_space),
        ("Live raw data", live_raw_space),
        ("Free-space managers", free_space_manager_space),
        ("Reusable free space", reusable_free_space),
        ("Unclassified", unclassified_space),
        ("Trailing slack", trailing_slack_space),
    ] {
        if extents.is_empty() {
            empty.push(category);
            continue;
        }
        extents.sort_unstable();
        for (start, length) in extents {
            writeln!(
                report,
                "  {category:<19} {start:>10} {:>10} {length:>10}",
                start.checked_add(length).unwrap()
            )
            .unwrap();
        }
    }
    if !empty.is_empty() {
        writeln!(report, "  No extents: {}", empty.join(", ")).unwrap();
    }
}
