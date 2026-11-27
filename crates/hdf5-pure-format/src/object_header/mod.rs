//! Object header parsing, message retention, and reference count observations for versions 1 and 2.

mod context;
mod filter;
mod header;
mod observation;
mod reference_count;
mod v1;
mod v2;

pub use filter::MessageFilter;
pub use header::{HeaderMessage, ObjectHeader};
pub use observation::ObjectHeaderChunk;
pub use reference_count::ObjectReferenceCount;
pub use reference_count::ObjectReferenceCountField;
pub use reference_count::ObjectReferenceCountIssue;
pub use reference_count::ObjectReferenceCountMessage;
pub use reference_count::ObjectReferenceCountObserver;
pub use reference_count::ObjectReferenceCountUpdate;
pub use reference_count::ObjectReferenceCountUpdateError;

pub use v2::AttributePhaseChange;
pub use v2::MessageBody;
pub use v2::MessageRecord;
pub use v2::MessageRecordLayout;
pub use v2::OBJECT_HEADER_PREFIX_MAX_LEN;
pub use v2::ObjectHeaderAllocation;
pub use v2::ObjectHeaderAllocationGeometry;
pub use v2::ObjectHeaderContinuation;
pub use v2::ObjectHeaderPrefix;
pub use v2::ObjectHeaderWriter;
pub use v2::ObjectTimes;
pub use v2::ParsedObjectHeaderPrefix;
pub use v2::PreparedMessageRecord;
pub use v2::PreparedRecordRef;
pub use v2::RecordSplice;
pub use v2::continuation_block_messages;

#[cfg(test)]
mod tests;
