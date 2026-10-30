use std::collections::HashMap;

use hdf5_pure::{AttrValue, Dataset, File, Group};

const MAX_GROUP_DEPTH: usize = 2;
const MAX_CHILDREN_PER_GROUP: usize = 8;

pub(crate) fn assert_same_file_metadata(buffered: &File, streaming: &File) {
    assert_eq!(buffered.file_size(), streaming.file_size());

    let left = buffered.superblock();
    let right = streaming.superblock();
    assert_eq!(left.version(), right.version());
    assert_eq!(left.offset_size, right.offset_size);
    assert_eq!(left.length_size, right.length_size);
    assert_eq!(left.base_address, right.base_address);
    assert_eq!(left.root_group_address, right.root_group_address);
    assert_eq!(left.eof_address, right.eof_address);
    assert_eq!(
        left.superblock_extension_address,
        right.superblock_extension_address
    );
}

pub(crate) fn assert_same_group(
    buffered_file: &File,
    streaming_file: &File,
    buffered_group: &Group,
    streaming_group: &Group,
    path: &str,
    depth: usize,
) {
    assert_same_attrs(buffered_group.attrs(), streaming_group.attrs());

    let buffered_datasets = buffered_group.datasets();
    let streaming_datasets = streaming_group.datasets();
    if let (Ok(buffered_datasets), Ok(streaming_datasets)) = (buffered_datasets, streaming_datasets)
    {
        assert_eq!(buffered_datasets, streaming_datasets);
        for name in buffered_datasets.iter().take(MAX_CHILDREN_PER_GROUP) {
            let buffered_dataset = buffered_group.dataset(name);
            let streaming_dataset = streaming_group.dataset(name);
            if let (Ok(buffered_dataset), Ok(streaming_dataset)) =
                (buffered_dataset, streaming_dataset)
            {
                assert_same_dataset(&buffered_dataset, &streaming_dataset);
            }

            let full_path = child_path(path, name);
            let buffered_dataset = buffered_file.dataset(&full_path);
            let streaming_dataset = streaming_file.dataset(&full_path);
            if let (Ok(buffered_dataset), Ok(streaming_dataset)) =
                (buffered_dataset, streaming_dataset)
            {
                assert_same_dataset(&buffered_dataset, &streaming_dataset);
            }
        }
    }

    if depth >= MAX_GROUP_DEPTH {
        return;
    }

    let buffered_groups = buffered_group.groups();
    let streaming_groups = streaming_group.groups();
    if let (Ok(buffered_groups), Ok(streaming_groups)) = (buffered_groups, streaming_groups) {
        assert_eq!(buffered_groups, streaming_groups);
        for name in buffered_groups.iter().take(MAX_CHILDREN_PER_GROUP) {
            let buffered_child = buffered_group.group(name);
            let streaming_child = streaming_group.group(name);
            if let (Ok(buffered_child), Ok(streaming_child)) = (buffered_child, streaming_child) {
                let full_path = child_path(path, name);
                assert_same_group(
                    buffered_file,
                    streaming_file,
                    &buffered_child,
                    &streaming_child,
                    &full_path,
                    depth + 1,
                );
            }
        }
    }
}

fn assert_same_dataset(buffered: &Dataset, streaming: &Dataset) {
    assert_same_result(buffered.shape(), streaming.shape());
    assert_same_result(buffered.dtype(), streaming.dtype());
    assert_same_attrs(buffered.attrs(), streaming.attrs());
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

fn assert_same_result<T: PartialEq>(
    buffered: Result<T, hdf5_pure::Error>,
    streaming: Result<T, hdf5_pure::Error>,
) {
    if let (Ok(buffered), Ok(streaming)) = (buffered, streaming) {
        assert!(buffered == streaming);
    }
}

/// Compare two attribute maps NaN-tolerantly and order-independently.
///
/// An attribute value may be a floating-point `NaN`, and `NaN != NaN` makes the
/// derived `PartialEq` on `AttrValue`/`HashMap` report a difference even when
/// both backends decoded the identical bytes. Debug-format each value (a `NaN`
/// renders as a stable `"NaN"`) and compare the sorted key/value pairs, so a
/// genuine divergence still fails while the `NaN` artifact does not.
fn assert_same_attrs(
    buffered: Result<HashMap<String, AttrValue>, hdf5_pure::Error>,
    streaming: Result<HashMap<String, AttrValue>, hdf5_pure::Error>,
) {
    if let (Ok(buffered), Ok(streaming)) = (buffered, streaming) {
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
        assert_eq!(left, right);
    }
}

fn child_path(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_owned()
    } else {
        format!("{parent}/{name}")
    }
}
