//! Compact attribute helpers for editor tests.

use super::{AttrOp, AttributePredecessors, CompactBatch, EditedAttrs, parse_compact_attr_name};
use crate::attribute_request::AttributeName;
use crate::edit::object_header::OhRegion;
use crate::error::{Error, FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::message_type::MessageType;
use crate::type_builders::AttrValue;

/// Applies compact edits for tests whose values require no global heap placement.
///
/// # Errors
///
/// Returns the parsing and operation errors of [`CompactBatch::parse`] and
/// [`CompactBatch::apply`]. Returns [`FormatError::Internal`] if the prepared result requires
/// heap placement. Propagates record-preparation errors from [`CompactBatch::into_compact`].
pub(in crate::edit) fn edit_compact_region(
    region: &OhRegion,
    ops: &[AttrOp],
) -> Result<OhRegion, Error> {
    let mut batch = CompactBatch::parse(region)?;
    for (order, operation) in ops.iter().enumerate() {
        batch.apply(operation, order)?;
    }
    let mut edit = batch.into_compact(AttributePredecessors::default())?;
    let EditedAttrs::RebuildCompact(compact) = edit.attrs else {
        return Err(FormatError::Internal("a compact batch returned dense storage".into()).into());
    };
    if !compact.fixups.is_empty() {
        return Err(
            FormatError::Internal("a compact test batch requires heap placement".into()).into(),
        );
    }
    edit.region.push_bytes(&compact.records);
    Ok(edit.region)
}

/// Returns the first matching attribute's creation index.
///
/// Returns `None` if the header is untracked or no matching record exists.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region or an Attribute
/// record has nonzero flags. Returns [`Error::Format`] if a name cannot be parsed.
pub(in crate::edit) fn attr_creation_index(
    region: &OhRegion,
    name: &str,
) -> Result<Option<u16>, Error> {
    if !region.layout().tracks_creation_order() {
        return Ok(None);
    }
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = region.next_message(p)? {
        if msg_type == MessageType::ATTRIBUTE
            && parse_compact_attr_name(region, p, body, body_end)? == name
        {
            return Ok(region.creation_index(p));
        }
        p = body_end;
    }
    Ok(None)
}

/// Serializes an inline value within the compact message-size limit.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the encoded message exceeds
/// [`OBJECT_HEADER_MESSAGE_MAX`]. Propagates attribute encoding errors.
///
/// # Panics
///
/// Panics in debug builds if `value` requires global heap storage.
pub(super) fn encode_attr_body(name: &str, value: &AttrValue) -> Result<Vec<u8>, Error> {
    debug_assert!(
        value.var_len_strings().is_none(),
        "a compact test value must not require global heap placement"
    );
    let body = AttributeName::try_from(name)?
        .build_value(value)?
        .serialize(
            crate::file_writer::OFFSET_WIDTH,
            crate::file_writer::LENGTH_SIZE,
        )?;
    if body.len() > OBJECT_HEADER_MESSAGE_MAX {
        return Err(Error::EditUnsupported(
            "group attribute is too large to encode in place",
        ));
    }
    Ok(body)
}
