//! Allocation tracking for filter unit tests.

use heapscope::Alloc;

#[global_allocator]
static ALLOC: Alloc = Alloc::system();
