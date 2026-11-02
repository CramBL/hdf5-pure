//! Admission and work accounting for version 1 B-tree metadata traversal.

use alloc::collections::BTreeSet;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// State for a depth-first traversal with unique metadata-node admission and a work limit.
///
/// Charges one unit for each admitted node and each queued child pointer. Leaf payload
/// addresses are outside this stack.
pub(crate) struct V1TreeVisitState {
    width: OffsetWidth,
    visited: BTreeSet<StoredAddress>,
    pending: Vec<PendingV1Node>,
    work: u64,
    limit: u64,
}

impl V1TreeVisitState {
    /// Starts a traversal with [`DEFAULT_WORK_LIMIT`] units of work.
    pub(crate) fn new(root: StoredAddress, width: OffsetWidth) -> Self {
        Self::with_limit(root, width, DEFAULT_WORK_LIMIT)
    }

    /// Starts a traversal with a caller-selected work limit.
    pub(crate) fn with_limit(root: StoredAddress, width: OffsetWidth, limit: u64) -> Self {
        Self {
            width,
            visited: BTreeSet::new(),
            pending: Vec::from([PendingV1Node {
                address: root,
                expected_level: None,
            }]),
            work: 0,
            limit,
        }
    }

    /// Admits the next metadata node before the caller reads it.
    ///
    /// Returns `None` when the stack is empty.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if the address exceeds the offset width,
    /// [`FormatError::InvalidBTreeSignature`] if it is undefined or already visited, and
    /// [`FormatError::ResourceLimit`] if admission exceeds the work limit.
    pub(crate) fn next(&mut self) -> Result<Option<AdmittedV1Node>, FormatError> {
        let Some(node) = self.pending.pop() else {
            return Ok(None);
        };
        let max = StoredAddress::undefined(self.width).get();
        if node.address.get() > max {
            return Err(FormatError::AddressTooLarge {
                address: node.address.get(),
                offset_size: self.width.get(),
            });
        }
        if node.address.is_undefined(self.width) || !self.visited.insert(node.address) {
            return Err(FormatError::InvalidBTreeSignature);
        }
        self.charge()?;
        Ok(Some(AdmittedV1Node {
            address: node.address,
            expected_level: node.expected_level,
        }))
    }

    /// Queues children with their parent's checked child level.
    ///
    /// The caller passes children in traversal order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ResourceLimit`] if queuing exceeds the work limit, or an error
    /// from `children`. Children queued before an error remain in the stack.
    pub(crate) fn push_children(
        &mut self,
        parent: V1InternalNode,
        children: impl Iterator<Item = Result<StoredAddress, FormatError>>,
    ) -> Result<(), FormatError> {
        let start = self.pending.len();
        for address in children {
            self.charge()?;
            self.pending.push(PendingV1Node {
                address: address?,
                expected_level: Some(parent.child_level),
            });
        }
        self.pending[start..].reverse();
        Ok(())
    }

    fn charge(&mut self) -> Result<(), FormatError> {
        let required = self.work.saturating_add(1);
        if self.work == u64::MAX || required > self.limit {
            return Err(FormatError::ResourceLimit {
                limit: self.limit,
                required,
            });
        }
        self.work = required;
        Ok(())
    }
}

struct PendingV1Node {
    address: StoredAddress,
    expected_level: Option<u8>,
}

/// A defined, representable metadata-node address admitted once for a traversal.
///
/// Holds the expected child level when the node has a parent.
#[derive(Debug)]
pub(crate) struct AdmittedV1Node {
    address: StoredAddress,
    expected_level: Option<u8>,
}

impl AdmittedV1Node {
    /// Returns the stored address to read.
    pub(crate) fn address(&self) -> StoredAddress {
        self.address
    }

    /// Checks the decoded level against the parent's expected child level.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeSignature`] if the levels differ.
    pub(crate) fn parse_level(self, level: u8) -> Result<V1NodeLevel, FormatError> {
        if self
            .expected_level
            .is_some_and(|expected| level != expected)
        {
            return Err(FormatError::InvalidBTreeSignature);
        }
        Ok(match level.checked_sub(1) {
            Some(child_level) => V1NodeLevel::Internal(V1InternalNode { child_level }),
            None => V1NodeLevel::Leaf,
        })
    }
}

/// A checked leaf level or an internal level with its expected child level.
pub(crate) enum V1NodeLevel {
    Internal(V1InternalNode),
    Leaf,
}

/// An internal node's checked child level.
pub(crate) struct V1InternalNode {
    child_level: u8,
}

/// The default allowance for admitted nodes and queued child pointers in one traversal.
const DEFAULT_WORK_LIMIT: u64 = 16_777_216;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::address::StoredAddressExt;

    #[rstest]
    #[case::root(0, 1)]
    #[case::edge(1, 2)]
    #[case::child(2, 3)]
    fn visit_work_exhaustion_returns_resource_limit(#[case] limit: u64, #[case] required: u64) {
        let mut visits =
            V1TreeVisitState::with_limit(StoredAddress::ZERO, OffsetWidth::Eight, limit);
        let result = (|| {
            let root = visits.next()?.unwrap();
            let V1NodeLevel::Internal(parent) = root.parse_level(1)? else {
                panic!("expected internal root");
            };
            visits.push_children(parent, core::iter::once(Ok(StoredAddress::new(64))))?;
            visits.next()?;
            Ok::<(), FormatError>(())
        })();
        assert_eq!(
            result.unwrap_err(),
            FormatError::ResourceLimit { limit, required }
        );
    }

    #[test]
    fn completed_work_at_limit_succeeds() {
        let mut visits = V1TreeVisitState::with_limit(StoredAddress::ZERO, OffsetWidth::Eight, 1);
        assert_eq!(
            visits.next().unwrap().unwrap().address(),
            StoredAddress::ZERO
        );
        assert!(visits.next().unwrap().is_none());
    }

    #[rstest]
    #[case::two(OffsetWidth::Two)]
    #[case::four(OffsetWidth::Four)]
    #[case::eight(OffsetWidth::Eight)]
    fn undefined_root_returns_invalid_btree_signature(#[case] width: OffsetWidth) {
        let mut visits = V1TreeVisitState::new(StoredAddress::undefined(width), width);
        let error = visits.next().unwrap_err();
        assert_eq!(error, FormatError::InvalidBTreeSignature);
    }

    #[rstest]
    #[case::two(OffsetWidth::Two)]
    #[case::four(OffsetWidth::Four)]
    fn unrepresentable_root_returns_address_too_large(#[case] width: OffsetWidth) {
        let address = StoredAddress::undefined(width).get() + 1;
        let mut visits = V1TreeVisitState::new(StoredAddress::new(address), width);
        let error = visits.next().unwrap_err();
        assert_eq!(
            error,
            FormatError::AddressTooLarge {
                address,
                offset_size: width.get()
            }
        );
    }
}
