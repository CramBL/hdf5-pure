//! Editing and append crosschecks against libhdf5.

mod append_inspection;
mod array_blocks;
mod attribute_collections;
mod attribute_storage;
mod edit_dense_links;
mod edit_tracked_dense_links;
mod extension_records;
mod file_widths;
mod finite_append;
mod group_link_inspection;
mod independent_widths;
mod large_page;
mod link_patches;
mod symbol_table_links;
mod version_1_attributes;

mod generic_large;
mod huge_links;
mod identity;

mod page_dense;
mod page_manager_accounting;
mod page_managers;
mod page_object_headers;
mod paged_index;
