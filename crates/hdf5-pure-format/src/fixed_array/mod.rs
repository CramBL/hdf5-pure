//! The Fixed Array chunk index, index type 3 of a version 4 data layout message.
//!
//! A Fixed Array indexes the chunks of a dataset whose maximum shape is fixed. Its header
//! (`FAHD`) points at one data block (`FADB`), which holds an element per slot. Past `2^page_bits`
//! slots the data block holds a page-init bitmap and then the elements in pages, each with its
//! own checksum.
//!
//! A caller reads the bytes of each range the parser returns and passes them back to it.
//! [`FixedArrayHeaderLocator`] returns the range of the header, [`FixedArrayDataBlockRef`] the
//! range of an unpaged block or of a paged block's prefix, page-init bitmap, and checksum.
//! [`FixedArrayPageRef`] gives the range of one page. [`FixedArrayUnpaged`] describes entry
//! windows and checksum ranges for an unpaged block.
//!
//! A writer plans the complete header and data block with [`FixedArrayPlan`] and supplies
//! their stored addresses to [`FixedArrayPlan::place`]. The data block includes all its pages.
//! The caller establishes allocation ownership and placement policy.
//!
//! The index is defined in "The Fixed Array Index" of the [format specification, version
//! 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_fixedarr

mod build;
mod data_block;
mod emit;
mod encode;
mod framing;
mod geometry;
mod header;
mod initialize;
mod page;
mod placement;
mod plan;

pub use build::build_fixed_array_at;
pub use build::fixed_array_len;
pub use data_block::FixedArrayDataBlock;
pub use data_block::FixedArrayDataBlockRef;
pub use data_block::FixedArrayElements;
pub use data_block::FixedArrayEntryWindow;
pub use data_block::FixedArrayUnpaged;
pub use header::FixedArrayHeader;
pub use header::FixedArrayHeaderLocator;
pub use page::FixedArrayPageBitmap;
pub use page::FixedArrayPageRef;
pub use placement::FixedArrayPlacement;
pub(crate) use plan::FIXED_ARRAY_PAGE_BITS;
pub use plan::FixedArrayPlan;

#[cfg(test)]
mod tests;
