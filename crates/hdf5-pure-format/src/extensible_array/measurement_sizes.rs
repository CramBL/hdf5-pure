//! Wire length formulas for Extensible Array blocks.

use super::BlockOffsetWidth;
use super::DataBlockGeometry;
use super::SuperBlockGeometry;
use crate::width::OffsetWidth;

/// Returns the length in bytes of an index block (`EAIB`): the prefix, `inline_elmts` elements,
/// the data block and super block addresses, and the checksum.
pub(crate) fn index_block_len(
    offset_width: OffsetWidth,
    inline_elmts: usize,
    elem_size: usize,
    ndblk_addrs: usize,
    nsblk_addrs: usize,
) -> usize {
    let os = usize::from(offset_width.get());
    4 + 1 + 1 + os // signature + version + client id + header address
        + inline_elmts * elem_size // inline element slots (always all written)
        + ndblk_addrs * os // direct data-block addresses
        + nsblk_addrs * os // super-block addresses
        + 4 // checksum
}

/// Returns the length in bytes of a data block (`EADB`) with the geometry `blocks`, paged or not.
pub(crate) fn data_block_len(
    blocks: DataBlockGeometry,
    elem_size: usize,
    offset_width: OffsetWidth,
    blk_off_size: BlockOffsetWidth,
) -> u64 {
    let prefix = (4 + 1 + 1 + u64::from(offset_width.get())) + blk_off_size.bytes() as u64;
    if blocks.is_paged() {
        // Paged: the prefix and its checksum, then every page and its checksum.
        prefix + 4 + blocks.npages() * (blocks.page_nelmts * elem_size as u64 + 4)
    } else {
        prefix + blocks.dblk_nelmts * elem_size as u64 + 4
    }
}

/// Returns the length in bytes of a super block (`EASB`): the prefix, the page-init bitmap of
/// paged data blocks, the data block addresses, and the checksum.
///
/// The length is computed in `u64`, so the counts are not narrowed to a 32-bit `usize` before
/// they are multiplied.
pub(crate) fn super_block_len(
    geom: SuperBlockGeometry,
    offset_width: OffsetWidth,
    blk_off_size: BlockOffsetWidth,
) -> u64 {
    let os = u64::from(offset_width.get());
    let header = 4 + 1 + 1 + os + blk_off_size.bytes() as u64;
    header + geom.bitmap_size() + geom.ndblks * os + 4
}
