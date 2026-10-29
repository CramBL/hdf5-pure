//! The global allocator for library tests.

use heapscope::Alloc;

#[global_allocator]
static ALLOC: Alloc = Alloc::system();
