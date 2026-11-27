//! The global heap collection parser and encoder.
//!
//! A collection holds global heap objects, such as the bytes of variable-length values, each at a
//! nonzero index. A global heap ID in the file holds the address of a collection and the index of
//! an object in it.
//!
//! Use [`GlobalHeapCollectionLocator`] to read the collection header and start a walk or a lookup.
//! To write a collection, add objects to a [`GlobalHeapCollectionWriter`] or supply their indices,
//! reference counts, and collection capacity to [`PreparedGlobalHeapCollection::new`].

mod builder;
mod collection;
mod compaction;
mod error;
mod index;
mod object;
mod read;
mod write;

pub use builder::GlobalHeapCollectionWriter;
pub use collection::{
    GLOBAL_HEAP_MIN_COLLECTION_SIZE, GlobalHeapCollection, GlobalHeapCollectionLocator,
};
pub use compaction::GlobalHeapCompaction;
pub use compaction::GlobalHeapFreeSpace;
pub use compaction::GlobalHeapMemberCollector;
pub use compaction::GlobalHeapMembers;
pub use compaction::GlobalHeapMove;
pub use error::GlobalHeapCollectionError;
pub use index::GLOBAL_HEAP_MAX_OBJECTS;
pub use object::{GlobalHeapObject, GlobalHeapPayloadRef};
pub use read::{GlobalHeapLookup, GlobalHeapWalk};
pub use write::{GlobalHeapMember, PreparedGlobalHeapCollection, encode_global_heap_collection};

#[cfg(test)]
mod tests;
