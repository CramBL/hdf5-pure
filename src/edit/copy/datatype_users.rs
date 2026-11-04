//! Uses of committed datatypes in prepared object copies.

use core::num::NonZeroU64;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use crate::address::StoredAddress;
use crate::attribute::AttributeMessage;
use crate::edit::object_header;
use crate::error::{Error, FormatError};
use crate::message_type::MessageType;
use crate::shared_message::{self, DatatypeLocation};
use crate::width::FormatWidths;

use super::CopyTree;

impl CopyTree {
    /// Counts the committed-datatype references retained by this copy.
    ///
    /// The caller passes the widths of the copy's source. Examines retained header records and
    /// normalized dense attributes. Returns `None` for unknown records, whole shared attribute
    /// records, object-header references in whole shared non-datatype records, or undefined
    /// targets. Returns `Some` with zero targets for a complete empty inventory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a header record is truncated. Propagates other
    /// header-record, attribute, and shared-reference framing errors. Returns [`Error::Format`]
    /// with [`FormatError::Internal`] if a use count overflows.
    ///
    /// # Performance
    ///
    /// Traverses each retained node and header record at most once. Allocates lookup storage for
    /// committed targets and retains one ordered map of distinct targets and positive
    /// multiplicities. Initial copy reads, owned copy storage, and value decoding are separate costs.
    pub(in crate::edit) fn datatype_users(
        &self,
        widths: FormatWidths,
    ) -> Result<Option<CopiedDatatypeUsers<'_>>, Error> {
        let Self { descendants, root } = self;
        let mut users = BTreeMap::new();
        for node in core::iter::once(root).chain(descendants) {
            for attr in node.dense_attrs().attrs() {
                if let DatatypeLocation::Committed(address) = attr.datatype_location {
                    if address.is_undefined(widths.offsets) {
                        return Ok(None);
                    }
                    add_user(&mut users, address)?;
                }
            }
            let region = node.region();
            let mut pos = 0;
            while let Some(record) = region
                .layout()
                .next_message(region, pos)
                .map_err(object_header::record_error)?
            {
                if !record.msg_type.is_known() {
                    return Ok(None);
                }
                let reference = match record.msg_type {
                    MessageType::DATATYPE if record.flags.is_shared() => Some(record.body),
                    MessageType::ATTRIBUTE if record.flags.is_shared() => return Ok(None),
                    MessageType::ATTRIBUTE => {
                        AttributeMessage::shared_datatype_reference(record.body)?
                    }
                    _ if record.flags.is_shared() => {
                        if shared_message::committed_address_in_with_widths(record.body, widths)?
                            .is_some()
                        {
                            return Ok(None);
                        }
                        None
                    }
                    _ => None,
                };
                if let Some(reference) = reference
                    && let Some(address) =
                        shared_message::committed_address_in_with_widths(reference, widths)?
                {
                    if address.is_undefined(widths.offsets) {
                        return Ok(None);
                    }
                    add_user(&mut users, address)?;
                }
                pos = record.body_range.end;
            }
        }
        Ok(Some(CopiedDatatypeUsers { tree: self, users }))
    }
}

/// Complete committed-datatype use counts associated with an immutable [`CopyTree`].
///
/// The original copy remains immutably borrowed for the summary's lifetime. Source freshness,
/// allocation ownership, and count-write permission require separate proofs.
pub(in crate::edit) struct CopiedDatatypeUsers<'copy> {
    tree: &'copy CopyTree,
    users: BTreeMap<StoredAddress, NonZeroU64>,
}

impl CopiedDatatypeUsers<'_> {
    /// Returns whether these counts were collected from `tree`.
    pub(in crate::edit) fn matches(&self, tree: &CopyTree) -> bool {
        core::ptr::eq(self.tree, tree)
    }

    /// Returns distinct stored targets and their positive multiplicities in address order.
    pub(in crate::edit) fn users(&self) -> impl Iterator<Item = (StoredAddress, NonZeroU64)> + '_ {
        self.users.iter().map(|(&address, &count)| (address, count))
    }
}

/// Adds one use of `address`, preserving positive multiplicities.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if the existing count overflows.
fn add_user(
    users: &mut BTreeMap<StoredAddress, NonZeroU64>,
    address: StoredAddress,
) -> Result<(), FormatError> {
    match users.entry(address) {
        Entry::Occupied(mut entry) => {
            let count = entry.get().checked_add(1).ok_or_else(|| {
                FormatError::Internal("copied datatype user count overflows".into())
            })?;
            entry.insert(count);
        }
        Entry::Vacant(entry) => {
            entry.insert(NonZeroU64::MIN);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
