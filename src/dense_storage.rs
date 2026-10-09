//! Shared ownership proof for version 2 B-tree indexes attached to dense storage.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hdf5_pure_space::__private::Extent;

use crate::address::StoredAddress;
use crate::btree_v2::{self, BTreeV2Header};
use crate::error::FormatError;
use crate::source::{Source, SourceMetadata};

/// One version 2 B-tree identified as part of a dense-storage owner's index set.
#[derive(Clone, Copy)]
pub(crate) struct DenseIndex {
    address: StoredAddress,
    expected_tree_type: u8,
}

impl DenseIndex {
    /// Pairs an index address with the client type its owner requires.
    pub(crate) const fn new(address: StoredAddress, expected_tree_type: u8) -> Self {
        Self {
            address,
            expected_tree_type,
        }
    }
}

/// Returns all allocations of a proven dense-storage index set.
///
/// Every index header must have the client type its owner requires, every tree must be enumerable
/// completely, the resulting allocations must be mutually disjoint, and no allocation may contain
/// the associated fractal-heap header. The caller decides which indexes exist from its own metadata
/// format and passes exactly that set here.
///
/// # Errors
///
/// Returns [`FormatError::InvalidBTreeNodeType`] for an unexpected client type, the structural
/// error from the version 2 B-tree storage walker for an incomplete tree, and
/// [`FormatError::InvalidBTreeV2Signature`] if the proven allocations overlap or contain
/// `heap_header`.
pub(crate) fn collect_dense_index_storage_extents<S: Source + ?Sized>(
    source: &S,
    heap_header: StoredAddress,
    indexes: impl IntoIterator<Item = DenseIndex>,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<Extent>, FormatError> {
    let mut extents = Vec::new();
    for index in indexes {
        let header = BTreeV2Header::parse_from_source(
            &SourceMetadata(source),
            index.address.get(),
            offset_size,
            length_size,
        )?;
        if header.tree_type != index.expected_tree_type {
            return Err(FormatError::InvalidBTreeNodeType(header.tree_type));
        }
        extents.extend(btree_v2::collect_btree_v2_storage_extents(
            source,
            index.address,
            offset_size,
            length_size,
        )?);
    }

    extents.sort_unstable();
    if extents
        .windows(2)
        .any(|pair| pair[0].end() > pair[1].start())
        || extents
            .iter()
            .any(|extent| extent.start() <= heap_header.get() && heap_header.get() < extent.end())
    {
        return Err(FormatError::InvalidBTreeV2Signature);
    }
    Ok(extents)
}

#[cfg(test)]
mod tests {
    use hdf5_pure_format::__private::{BTREE_V2_LINK_CREATION_ORDER, BTREE_V2_LINK_NAME};
    use test_util::btree_v2;
    use test_util::widths::Widths;

    use super::*;
    use crate::source::BytesSource;

    const WIDTHS: Widths = Widths::EIGHT;
    const NODE_SIZE: u64 = 512;
    const RECORD_SIZE: u16 = 16;

    fn place_index_tree(
        file: &mut [u8],
        header_at: usize,
        root_at: usize,
        tree_type: u8,
    ) -> Vec<Extent> {
        let record = vec![0; usize::from(RECORD_SIZE)];
        let leaf = btree_v2::leaf(tree_type, &[record]);
        let header =
            btree_v2::Header::new(tree_type, RECORD_SIZE, u64::try_from(root_at).unwrap(), 1)
                .build(WIDTHS);
        file[header_at..header_at + header.len()].copy_from_slice(&header);
        file[root_at..root_at + leaf.len()].copy_from_slice(&leaf);
        vec![
            Extent::new(
                u64::try_from(header_at).unwrap(),
                u64::try_from(header.len()).unwrap(),
            )
            .unwrap(),
            Extent::new(u64::try_from(root_at).unwrap(), NODE_SIZE).unwrap(),
        ]
    }

    #[test]
    fn collects_name_and_creation_order_indexes() {
        let mut file = vec![0; 0x1000];
        let mut expected = place_index_tree(&mut file, 0x100, 0x200, BTREE_V2_LINK_NAME);
        expected.extend(place_index_tree(
            &mut file,
            0x500,
            0x600,
            BTREE_V2_LINK_CREATION_ORDER,
        ));
        expected.sort_unstable();

        let extents = collect_dense_index_storage_extents(
            &BytesSource::new(file),
            StoredAddress::new(0x80),
            [
                DenseIndex::new(StoredAddress::new(0x100), BTREE_V2_LINK_NAME),
                DenseIndex::new(StoredAddress::new(0x500), BTREE_V2_LINK_CREATION_ORDER),
            ],
            8,
            8,
        )
        .unwrap();

        assert_eq!(extents, expected);
    }

    #[test]
    fn rejects_an_unexpected_client_type() {
        let mut file = vec![0; 0x800];
        place_index_tree(&mut file, 0x100, 0x200, BTREE_V2_LINK_CREATION_ORDER);

        let err = collect_dense_index_storage_extents(
            &BytesSource::new(file),
            StoredAddress::new(0x80),
            [DenseIndex::new(
                StoredAddress::new(0x100),
                BTREE_V2_LINK_NAME,
            )],
            8,
            8,
        )
        .unwrap_err();

        assert_eq!(
            err,
            FormatError::InvalidBTreeNodeType(BTREE_V2_LINK_CREATION_ORDER)
        );
    }

    #[test]
    fn rejects_overlapping_index_allocations() {
        let mut file = vec![0; 0x800];
        place_index_tree(&mut file, 0x100, 0x200, BTREE_V2_LINK_NAME);
        let index = DenseIndex::new(StoredAddress::new(0x100), BTREE_V2_LINK_NAME);

        let err = collect_dense_index_storage_extents(
            &BytesSource::new(file),
            StoredAddress::new(0x80),
            [index, index],
            8,
            8,
        )
        .unwrap_err();

        assert_eq!(err, FormatError::InvalidBTreeV2Signature);
    }

    #[test]
    fn rejects_an_index_containing_the_heap_header() {
        let mut file = vec![0; 0x800];
        place_index_tree(&mut file, 0x100, 0x200, BTREE_V2_LINK_NAME);

        let err = collect_dense_index_storage_extents(
            &BytesSource::new(file),
            StoredAddress::new(0x100),
            [DenseIndex::new(
                StoredAddress::new(0x100),
                BTREE_V2_LINK_NAME,
            )],
            8,
            8,
        )
        .unwrap_err();

        assert_eq!(err, FormatError::InvalidBTreeV2Signature);
    }

    #[test]
    fn a_truncated_later_index_fails_the_set_atomically() {
        let mut file = vec![0; 0x900];
        place_index_tree(&mut file, 0x100, 0x200, BTREE_V2_LINK_NAME);
        let root_at = 0x880usize;
        let header = btree_v2::Header::new(
            BTREE_V2_LINK_CREATION_ORDER,
            RECORD_SIZE,
            u64::try_from(root_at).unwrap(),
            1,
        )
        .build(WIDTHS);
        file[0x500..0x500 + header.len()].copy_from_slice(&header);
        let available = file.len();

        let err = collect_dense_index_storage_extents(
            &BytesSource::new(file),
            StoredAddress::new(0x80),
            [
                DenseIndex::new(StoredAddress::new(0x100), BTREE_V2_LINK_NAME),
                DenseIndex::new(StoredAddress::new(0x500), BTREE_V2_LINK_CREATION_ORDER),
            ],
            8,
            8,
        )
        .unwrap_err();

        assert_eq!(
            err,
            FormatError::UnexpectedEof {
                expected: root_at + usize::try_from(NODE_SIZE).unwrap(),
                available,
            }
        );
    }
}
