//! Version 2 object header prefixes, message records, and continuation blocks.
//!
//! [`MessageRecordLayout::prepare_record`] prepares a typed inline record through [`MessageBody`].
//! [`MessageRecordLayout::prepare_raw_record`] borrows an encoded body, including a shared message
//! reference. Write the returned [`PreparedMessageRecord`] into a slice or append it to a buffer.

mod allocation;
mod chunk;
mod collector;
mod continuation;
mod prefix;
mod prepared_record;
mod record;
mod splice;
mod traversal;
mod worklist;
mod writer;

pub use allocation::{ObjectHeaderAllocation, ObjectHeaderAllocationGeometry};
pub(super) use chunk::CHECKSUM_LEN;
pub use continuation::{ObjectHeaderContinuation, continuation_block_messages};
pub use prefix::{
    AttributePhaseChange, OBJECT_HEADER_PREFIX_MAX_LEN, ObjectHeaderPrefix, ObjectTimes,
    ParsedObjectHeaderPrefix,
};
pub use prepared_record::{MessageBody, PreparedMessageRecord, PreparedRecordRef};
pub use record::{MessageRecord, MessageRecordLayout};
pub use splice::RecordSplice;
pub use writer::ObjectHeaderWriter;
