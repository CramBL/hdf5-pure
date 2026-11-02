//! Persistent file fixtures with separated raw-data holes for staged appends.

use std::ops::Range;
use std::path::Path;

use hdf5_pure::{AttrValue, File, FileBuilder, FileSpaceSettings, FileSpaceStrategy};

use crate::dataset::Unlimited;
use crate::session;

use super::persisted::PersistedStrategy;

/// Returns an editing session and the physical ranges of retired scratch data.
///
/// Creates a persistent file under `strategy` with a chunked dataset `d` containing
/// [`CHUNK_ELEMENTS`] values. Creates alternating scratch and spacer datasets, then commits
/// deletion of the scratch datasets. Each returned range describes one scratch dataset's
/// complete raw allocation before deletion. Retains the spacer datasets.
///
/// For PAGE allocation, retains `metadata_guard` and deletes `metadata_reserve`, each with a
/// [`METADATA_BYTES`]-byte padding attribute. The returned [`File`] remains open for appends.
/// The caller closes it before reading persisted bytes through another file handle.
///
/// # Panics
///
/// Panics if creating, opening, inspecting, editing, or committing the fixture fails.
///
/// # Examples
///
/// ```
/// use test_util_hdf5::file_space::chunk_holes;
/// use test_util_hdf5::file_space::persisted::PersistedStrategy;
///
/// let directory = tempfile::tempdir()?;
/// let path = directory.path().join("holes.h5");
/// let (file, holes) = chunk_holes::prepare(&path, PersistedStrategy::Page);
/// assert_eq!(holes.len(), chunk_holes::APPENDED_CHUNKS);
/// assert!(holes.iter().all(|range| range.start < range.end));
/// file.close()?;
/// directory.close()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn prepare(path: &Path, strategy: PersistedStrategy) -> (File, Vec<Range<u64>>) {
    let paged = matches!(strategy, PersistedStrategy::Page);
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(match strategy {
            PersistedStrategy::FsmAggr => FileSpaceStrategy::FsmAggr,
            PersistedStrategy::Page => FileSpaceStrategy::Page,
        })
        .with_persistence(true)
        .with_threshold(0),
    );
    Unlimited::new(
        "d",
        &(0..CHUNK_ELEMENTS as i32).collect::<Vec<_>>(),
        CHUNK_ELEMENTS as u64,
    )
    .add_to(&mut builder);
    builder.write(path).unwrap();
    let file = session::open_rw(path).unwrap();
    for index in 0..APPENDED_CHUNKS {
        for role in ["scratch", "spacer"] {
            file.root()
                .create_dataset(&format!("{role}{index}"), |builder| {
                    builder.with_i32_data(&(0..72).collect::<Vec<i32>>());
                })
                .unwrap();
        }
    }
    file.commit().unwrap();
    if paged {
        for role in ["metadata_guard", "metadata_reserve"] {
            file.root()
                .create_group_with(role, |group| {
                    group.set_attr("padding", AttrValue::U8Array(vec![0; METADATA_BYTES]));
                })
                .unwrap();
        }
        file.commit().unwrap();
    }
    let holes = (0..APPENDED_CHUNKS)
        .map(|index| super::contiguous_extent(&file, &format!("scratch{index}")))
        .collect();
    for index in 0..APPENDED_CHUNKS {
        file.root().delete(&format!("scratch{index}")).unwrap();
    }
    if paged {
        file.root().delete("metadata_reserve").unwrap();
    }
    file.commit().unwrap();
    (file, holes)
}

/// The number of retired scratch datasets and chunks in the tested append.
pub const APPENDED_CHUNKS: usize = 16;
/// The number of [`i32`] values per chunk in dataset `d`.
pub const CHUNK_ELEMENTS: usize = 64;
/// The padding attribute length in each PAGE metadata group, in bytes.
pub const METADATA_BYTES: usize = 1_536;

#[cfg(test)]
mod tests {
    use hdf5_pure::{File, FreeSpaceManagerRole};
    use test_util::range;

    use crate::file_space::chunk_holes;
    use crate::file_space::persisted::PersistedStrategy;
    use crate::session;

    #[test]
    fn page_fixture_keeps_metadata_headroom_and_fragmented_raw_capacity() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("holes.h5");
        let (file, original_holes) = chunk_holes::prepare(&path, PersistedStrategy::Page);
        let saved = file
            .reusable_free_sections()
            .unwrap()
            .sections()
            .iter()
            .map(|section| (section.address().get(), section.length().get()))
            .collect::<Vec<_>>();
        let size = file.file_size();
        file.close().unwrap();
        let snapshot = File::from_bytes(std::fs::read(&path).unwrap()).unwrap();
        let recovered = session::open_rw(&path).unwrap();
        assert_eq!(recovered.file_size(), size);
        assert_eq!(
            recovered
                .reusable_free_sections()
                .unwrap()
                .sections()
                .iter()
                .map(|section| (section.address().get(), section.length().get()))
                .collect::<Vec<_>>(),
            saved
        );
        recovered.close().unwrap();
        let persisted = snapshot.persisted_free_sections().unwrap().unwrap();
        assert!(
            persisted
                .managers()
                .iter()
                .filter(|manager| manager.role() == FreeSpaceManagerRole::PageSmallMetadata)
                .any(|manager| manager
                    .free_sections()
                    .sections()
                    .iter()
                    .any(|section| section.length().get() >= chunk_holes::METADATA_BYTES as u64))
        );
        let payload = (chunk_holes::CHUNK_ELEMENTS
            * chunk_holes::APPENDED_CHUNKS
            * std::mem::size_of::<i32>()) as u64;
        let raw_sections = persisted
            .managers()
            .iter()
            .filter(|manager| {
                matches!(
                    manager.role(),
                    FreeSpaceManagerRole::PageSmallRawData | FreeSpaceManagerRole::PageLargeGeneric
                )
            })
            .flat_map(|manager| manager.free_sections().sections())
            .map(|section| {
                (
                    snapshot.resolve_address(section.address()).unwrap().get(),
                    section.length().get(),
                )
            })
            .collect::<Vec<_>>();
        assert!(
            raw_sections.iter().all(|&(_, length)| length < payload),
            "{raw_sections:?}"
        );
        let eligible = saved
            .iter()
            .copied()
            .filter(|&(start, length)| {
                range::covered_len(&raw_sections, &(start..start + length)) == length
            })
            .collect::<Vec<_>>();
        assert!(
            eligible.iter().map(|&(_, length)| length).sum::<u64>() >= payload,
            "{eligible:?}"
        );
        assert!(
            eligible.iter().all(|&(_, length)| length < payload),
            "{eligible:?}"
        );
        let selected = saved
            .iter()
            .copied()
            .filter(|&(start, length)| {
                original_holes
                    .iter()
                    .any(|hole| start < hole.end && hole.start < start + length)
            })
            .collect::<Vec<_>>();
        for &(start, length) in &selected {
            assert_eq!(
                range::covered_len(&raw_sections, &(start..start + length)),
                length
            );
        }
        for hole in original_holes {
            assert_eq!(range::covered_len(&eligible, &hole), hole.end - hole.start);
        }
        directory.close().unwrap();
    }
}
