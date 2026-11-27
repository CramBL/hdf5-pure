use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;
use test_util::global_heap;
use test_util::widths::Widths;

use crate::error::FormatError;
use crate::global_heap::{GlobalHeapCollection, GlobalHeapCollectionLocator};
use crate::width::LengthWidth;

pub(super) fn region(start: u64, bytes: &[u8]) -> FileRegion<'_> {
    FileRegion::new(
        FileRange::new(PhysicalAddress::new(start), bytes.len() as u64).unwrap(),
        bytes,
    )
    .unwrap()
}

/// Parses the header of the collection at the start of `image`.
pub(super) fn collection_at(image: FileRegion<'_>, lengths: LengthWidth) -> GlobalHeapCollection {
    let locator = GlobalHeapCollectionLocator::new(image.range().start(), lengths);
    locator
        .parse(image.subregion(locator.parse_range().unwrap()).unwrap())
        .unwrap()
}

/// Walks the collection at the start of `image`, and returns the index and the data of each
/// object.
pub(super) fn walk_objects(
    image: FileRegion<'_>,
    lengths: LengthWidth,
) -> Result<Vec<(u16, &[u8])>, FormatError> {
    let mut walk = collection_at(image, lengths).walk();
    let mut objects = Vec::new();
    while let Some(range) = walk.parse_range() {
        if let Some(object) = walk.parse(image.subregion(range)?)? {
            objects.push((object.index().get(), object.payload().parse(image)?));
        }
    }
    Ok(objects)
}

/// Returns a collection of `size` bytes that holds `objects`, each at its index with a
/// reference count of 1, and a free space object where the tail has room for its header.
pub(super) fn padded_collection(
    lengths: LengthWidth,
    size: usize,
    objects: &[(u16, &[u8])],
) -> Vec<u8> {
    let objects: Vec<_> = objects
        .iter()
        .map(|&(index, data)| global_heap::Object::new(index, data))
        .collect();
    global_heap::collection_of_size(&objects, Widths::new(8, usize::from(lengths.get())), size)
}
