//! The Extensible Array chunk index, index type 4 of a version 4 data layout message.
//!
//! An Extensible Array indexes the chunks of a dataset with one unlimited dimension. Its header
//! (`EAHD`) points at an index block (`EAIB`), which holds the first elements, the addresses of
//! the first data blocks (`EADB`), and the addresses of super blocks (`EASB`), each of which
//! addresses further data blocks. A data block of more than one page of elements is paged, and
//! the super block that addresses it holds its page-init bitmap. The index is defined in "The
//! Extensible Array Index" of the [format specification, version 4.0][spec], which calls a super
//! block a secondary block.
//!
//! A caller reads the bytes of each range the parser returns and passes them back to it.
//! [`ExtensibleArrayHeaderLocator`] returns the range of the header, and each reference to a block
//! or a page, from [`ExtensibleArrayIndexBlockRef`] to [`ExtensibleArrayPageRef`], the range a
//! caller reads for it. The reference to a super block or a data block also returns whether the
//! block holds a slot below the header's "Max Index Set", one more than the highest slot the array
//! has set.
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr

#[cfg(test)]
mod measurement_sizes;

mod block;
mod creation;
mod data_block;
mod elements;
mod geometry;
mod header;
mod index_block;
mod page;
mod secondary_block;
mod statistics;
pub(crate) mod write;

pub use block::ExtensibleArrayBlockFraming;
pub use creation::{
    BlockOffsetWidth, EA_CLIENT_ID_UNFILTERED, ExtensibleArrayBitGeometry,
    ExtensibleArrayCreationParameters, MaxIndexBits, PageElementCount, extensible_array_capacity,
};
pub use data_block::{ExtensibleArrayDataBlock, ExtensibleArrayDataBlockRef};
pub use elements::ExtensibleArrayElements;
pub use geometry::{DataBlockGeometry, ExtensibleArrayGeometry, SuperBlockGeometry};
pub use header::{ExtensibleArray, ExtensibleArrayHeader, ExtensibleArrayHeaderLocator};
pub use index_block::{ExtensibleArrayIndexBlock, ExtensibleArrayIndexBlockRef};
pub use page::{ExtensibleArrayPageRef, ExtensibleArrayPages};
pub use secondary_block::{ExtensibleArraySuperBlock, ExtensibleArraySuperBlockRef};
pub use statistics::ExtensibleArrayStats;
pub use write::{
    ExtensibleArrayInitialization, ExtensibleArrayLayout, SlotOccupancy, build_extensible_array_at,
    extensible_array_layout, extensible_array_len,
};

#[cfg(test)]
mod tests;
