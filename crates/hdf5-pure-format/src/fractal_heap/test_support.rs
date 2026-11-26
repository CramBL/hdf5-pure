use super::header::FractalHeapFiltering;
use super::header::FractalHeapHeader;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::filter_pipeline::FilterDescription;
use crate::filter_pipeline::FilterPipeline;

/// Returns a header with the supplied doubling-table parameters.
pub(super) fn dtable_header(
    start_block_size: u64,
    max_direct_block_size: u64,
    table_width: u16,
) -> FractalHeapHeader {
    FractalHeapHeader {
        heap_id_length: 7,
        filtering: FractalHeapFiltering::Unfiltered,
        flags: 0,
        max_managed_object_size: 0,
        next_huge_object_id: 0,
        btree_huge_objects_address: StoredAddress::new(u64::MAX),
        free_space_in_managed_blocks: 0,
        managed_block_free_space_manager_address: StoredAddress::new(u64::MAX),
        managed_space: 0,
        allocated_managed_space: 0,
        direct_block_allocation_iterator_offset: 0,
        managed_objects_count: 0,
        huge_objects_size: 0,
        huge_objects_count: 0,
        tiny_objects_size: 0,
        tiny_objects_count: 0,
        table_width,
        starting_block_size: start_block_size,
        max_direct_block_size,
        max_heap_size: 40,
        start_root_rows: 1,
        root_block_address: StoredAddress::new(0),
        current_rows_in_root_indirect_block: 0,
    }
}
pub(super) fn filtered_heap() -> FractalHeapFiltering {
    FractalHeapFiltering::Filtered {
        root_direct_block_size: 91,
        root_filter_mask: 1,
        pipeline: FilterPipeline::new(
            PIPELINE_VERSION_TWO,
            vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: vec![6],
            }],
        )
        .unwrap(),
    }
}
// "The Data Storage - Filter Pipeline Message", format specification version 4.0, defines
// versions 1 and 2 and deflate ID 1.
pub(super) const PIPELINE_VERSION_ONE: u8 = 1;
pub(super) const PIPELINE_VERSION_TWO: u8 = 2;
pub(super) const FILTER_DEFLATE: u16 = 1;

/// Returns a header with the heap ID layout of the attribute heaps `hdf5-pure` writes: 8-byte
/// IDs over a 40-bit heap, with a huge-object B-tree.
pub(super) fn attribute_heap_header() -> FractalHeapHeader {
    FractalHeapHeader {
        heap_id_length: 8,
        max_heap_size: 40,
        max_managed_object_size: 65_535,
        btree_huge_objects_address: StoredAddress::new(0x800),
        ..dtable_header(1024, 65536, 4)
    }
}
