//! Buffered and streaming metadata comparison shared by fuzzing and regression tests.
//!
//! Both readers open the same file path. Outcomes are compared at corresponding operations,
//! with one classification for backend-specific failures outside the input range.
//!
//! Visits descend through at most two subgroup levels and select at most eight dataset names
//! and eight subgroup names per visited group. Name lists are collected and compared in full
//! before child visits are limited.

use core::fmt::Debug;
use core::mem;
use core::mem::Discriminant;
use std::collections::HashMap;

use hdf5_pure::{AttrValue, Dataset, Error, File, FormatError, Group};
use tempfile::NamedTempFile;

/// Compares both readers using a temporary file containing `data`.
///
/// # Panics
///
/// Panics if fixture I/O fails, either reader reports an internal failure, or reader outcomes
/// or metadata differ beyond input-range normalization.
pub(crate) fn compare(data: &[u8]) -> Comparison {
    let path = NamedTempFile::new()
        .unwrap_or_else(|error| panic!("create differential fixture: {error}"))
        .into_temp_path();
    std::fs::write(&path, data)
        .unwrap_or_else(|error| panic!("write differential fixture {path:?}: {error}"));
    let Some((buffered, streaming)) = matched(
        Operation::FileOpen,
        File::open(&path),
        File::open_streaming(&path),
    ) else {
        return Comparison::Rejected;
    };
    assert_eq!(buffered.file_size(), streaming.file_size());
    assert_eq!(buffered.superblock(), streaming.superblock());
    let mut counts = Counts {
        groups: 0,
        datasets: 0,
    };
    compare_group(
        &buffered,
        &streaming,
        &buffered.root(),
        &streaming.root(),
        "",
        0,
        &mut counts,
    );
    Comparison::Opened {
        groups: counts.groups,
        datasets: counts.datasets,
    }
}

fn compare_group(
    buffered_file: &File,
    streaming_file: &File,
    buffered_group: &Group,
    streaming_group: &Group,
    path: &str,
    depth: usize,
    counts: &mut Counts,
) {
    counts.groups += 1;
    compare_attrs(
        Operation::GroupAttributes,
        buffered_group.attrs(),
        streaming_group.attrs(),
    );
    if let Some((left, right)) = matched(
        Operation::DatasetNames,
        buffered_group.datasets(),
        streaming_group.datasets(),
    ) {
        assert_eq!(left, right, "dataset names at {path:?}");
        for name in left.iter().take(MAX_CHILDREN_PER_GROUP) {
            if let Some((buffered, streaming)) = matched(
                Operation::RelativeDataset,
                buffered_group.dataset(name),
                streaming_group.dataset(name),
            ) {
                counts.datasets += 1;
                compare_dataset(&buffered, &streaming);
            }
            let full_path = child_path(path, name);
            if let Some((buffered, streaming)) = matched(
                Operation::AbsoluteDataset,
                buffered_file.dataset(&full_path),
                streaming_file.dataset(&full_path),
            ) {
                counts.datasets += 1;
                compare_dataset(&buffered, &streaming);
            }
        }
    }
    if depth >= MAX_GROUP_DEPTH {
        return;
    }
    if let Some((left, right)) = matched(
        Operation::GroupNames,
        buffered_group.groups(),
        streaming_group.groups(),
    ) {
        assert_eq!(left, right, "group names at {path:?}");
        for name in left.iter().take(MAX_CHILDREN_PER_GROUP) {
            if let Some((buffered, streaming)) = matched(
                Operation::Subgroup,
                buffered_group.group(name),
                streaming_group.group(name),
            ) {
                compare_group(
                    buffered_file,
                    streaming_file,
                    &buffered,
                    &streaming,
                    &child_path(path, name),
                    depth + 1,
                    counts,
                );
            }
        }
    }
}

fn compare_dataset(buffered: &Dataset, streaming: &Dataset) {
    compare_value(Operation::Shape, buffered.shape(), streaming.shape());
    compare_value(Operation::Datatype, buffered.dtype(), streaming.dtype());
    compare_attrs(
        Operation::DatasetAttributes,
        buffered.attrs(),
        streaming.attrs(),
    );
    assert_eq!(
        buffered.chunk_cache_stats().index_loaded(),
        streaming.chunk_cache_stats().index_loaded()
    );
    assert_eq!(
        buffered.chunk_cache_stats().cached_chunks(),
        streaming.chunk_cache_stats().cached_chunks()
    );
    assert_eq!(
        buffered.chunk_cache_stats().cached_bytes(),
        streaming.chunk_cache_stats().cached_bytes()
    );
}

pub(crate) fn compare_value<T: Debug + PartialEq>(
    operation: Operation,
    buffered: Result<T, Error>,
    streaming: Result<T, Error>,
) {
    if let Some((left, right)) = matched(operation, buffered, streaming) {
        assert_eq!(left, right, "{operation:?}: different values");
    }
}

/// Returns both success values, or [`None`] for matching failures.
///
/// # Panics
///
/// Panics if outcomes disagree or either reader reports an infrastructure or internal error.
pub(crate) fn matched<T>(
    operation: Operation,
    buffered: Result<T, Error>,
    streaming: Result<T, Error>,
) -> Option<(T, T)> {
    match (buffered, streaming) {
        (Ok(left), Ok(right)) => Some((left, right)),
        (Err(left), Err(right)) => {
            assert_eq!(
                Failure::of(operation, &left),
                Failure::of(operation, &right),
                "{operation:?}: different failures"
            );
            None
        }
        (Ok(_), Err(error)) => {
            panic!("{operation:?}: buffered succeeded but streaming rejected: {error:?}")
        }
        (Err(error), Ok(_)) => {
            panic!("{operation:?}: buffered rejected but streaming succeeded: {error:?}")
        }
    }
}

/// Compares outcomes and successful attribute maps using sorted keys and [`Debug`] output.
///
/// NaN bit patterns compare equally within each attribute type. Map iteration order is ignored.
pub(crate) fn compare_attrs(
    operation: Operation,
    buffered: Result<HashMap<String, AttrValue>, Error>,
    streaming: Result<HashMap<String, AttrValue>, Error>,
) {
    if let Some((buffered, streaming)) = matched(operation, buffered, streaming) {
        let mut left: Vec<(String, String)> = buffered
            .iter()
            .map(|(k, v)| (k.clone(), format!("{v:?}")))
            .collect();
        let mut right: Vec<(String, String)> = streaming
            .iter()
            .map(|(k, v)| (k.clone(), format!("{v:?}")))
            .collect();
        left.sort();
        right.sort();
        assert_eq!(left, right, "{operation:?}: different values");
    }
}

fn child_path(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_owned()
    } else {
        format!("{parent}/{name}")
    }
}

/// Matching file-open rejection or counts from the metadata comparison.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum Comparison {
    Opened {
        /// Group visits, including the root.
        groups: usize,
        /// Successful dataset opens, counting relative and root-relative lookups separately.
        datasets: usize,
    },
    Rejected,
}

/// The reader operation named in a comparison failure.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Operation {
    AbsoluteDataset,
    DatasetAttributes,
    DatasetNames,
    Datatype,
    FileOpen,
    GroupAttributes,
    GroupNames,
    RelativeDataset,
    Shape,
    Subgroup,
}

/// Comparable reader failures after input-range normalization.
#[derive(Debug, Eq, PartialEq)]
enum Failure<'a> {
    File {
        kind: Discriminant<Error>,
        description: String,
    },
    Format(&'a FormatError),
    /// Insufficient region coverage, buffer EOF, source EOF, or an overflowing range end.
    OutsideInput,
}

impl<'a> Failure<'a> {
    fn of(operation: Operation, error: &'a Error) -> Self {
        match error {
            Error::Io(_) | Error::Format(FormatError::Source(_)) => {
                panic!("{operation:?}: infrastructure failure: {error:?}")
            }
            Error::Format(FormatError::Internal(_)) => {
                panic!("{operation:?}: internal failure: {error:?}")
            }
            Error::Format(
                FormatError::InsufficientRegion { .. }
                | FormatError::UnexpectedEof { .. }
                | FormatError::UnexpectedSourceEof { .. },
            ) => Self::OutsideInput,
            Error::Format(FormatError::OffsetOverflow { offset, length })
                if offset.checked_add(*length).is_none() =>
            {
                Self::OutsideInput
            }
            Error::Format(error) => Self::Format(error),
            _ => Self::File {
                kind: mem::discriminant(error),
                description: format!("{error:?}"),
            },
        }
    }
}

struct Counts {
    groups: usize,
    datasets: usize,
}

const MAX_GROUP_DEPTH: usize = 2;
const MAX_CHILDREN_PER_GROUP: usize = 8;
