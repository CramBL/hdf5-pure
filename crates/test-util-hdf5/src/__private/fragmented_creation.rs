use std::path::Path;

use hdf5_pure::MaxExtent;

use crate::file_space::chunk_holes;
use crate::file_space::persisted::PersistedStrategy;

/// Creates `replacement` in existing free extents and returns its element values.
///
/// Checks that the sixteen chunks occupy original free extents, each smaller than the complete
/// payload. Uses an unlimited dimension when `extensible` is `true`. Closes the file before
/// returning, retaining the seed dataset and spacers for survivor checks.
///
/// # Panics
///
/// Panics if fixture creation or I/O fails, or placement and value checks fail.
pub fn write(path: &Path, strategy: PersistedStrategy, extensible: bool) -> Vec<i32> {
    let (file, _) = chunk_holes::prepare(path, strategy);
    let free = file
        .reusable_free_sections()
        .unwrap()
        .sections()
        .iter()
        .map(|section| (section.address().get(), section.length().get()))
        .collect::<Vec<_>>();
    let expected = (0..1_024).collect::<Vec<i32>>();
    let dataset = file
        .root()
        .create_dataset("replacement", |builder| {
            builder.with_i32_data(&expected).with_chunks(&[64]);
            if extensible {
                builder.with_maxshape(&[MaxExtent::Unlimited]);
            }
        })
        .unwrap();
    file.commit().unwrap();
    let chunks = dataset.chunks().unwrap();
    assert_eq!(chunks.len(), 16);
    assert!(free.iter().all(|&(_, length)| length < 4_096));
    assert!(chunks.iter().all(
        |chunk| free.iter().any(|&(start, length)| start <= chunk.address
            && chunk.address + chunk.storage_size <= start + length)
    ));
    assert_eq!(dataset.read_i32().unwrap(), expected);
    drop(dataset);
    file.close().unwrap();
    expected
}
