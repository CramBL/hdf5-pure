//! Iterative full-tree traversal with distinct node addresses and a work limit.

#[cfg(test)]
use core::cell::Cell;

use alloc::collections::BTreeSet;
#[cfg(test)]
use alloc::rc::Rc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hdf5_pure_format::__private::{BTreeV2Node, BTreeV2NodeView, BTreeV2Record};

use crate::address::StoredAddress;
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// Visited addresses, pending steps, and a work allowance for one full-tree traversal.
///
/// The reader checks each node address before acquiring its bytes and charges one unit of work
/// for each node visit and child edge. It keeps pending records and child nodes on an explicit
/// stack and releases each node's bytes before acquiring the next node.
pub(super) struct V2TreeVisitState<'layout> {
    width: OffsetWidth,
    visited: BTreeSet<StoredAddress>,
    pending: Vec<V2VisitStep<'layout>>,
    kind: V2WalkKind,
    work: u64,
    limit: u64,
    #[cfg(test)]
    copied_bytes: Option<Rc<Cell<usize>>>,
}

impl<'layout> V2TreeVisitState<'layout> {
    /// Creates a walk with the default work allowance.
    pub(super) fn new(root: BTreeV2Node<'layout>, width: OffsetWidth, kind: V2WalkKind) -> Self {
        Self::with_limit(root, width, kind, DEFAULT_WORK_LIMIT)
    }

    /// Creates a walk with at most `limit` node visits and child edges combined.
    pub(super) fn with_limit(
        root: BTreeV2Node<'layout>,
        width: OffsetWidth,
        kind: V2WalkKind,
        limit: u64,
    ) -> Self {
        Self {
            width,
            visited: BTreeSet::new(),
            pending: Vec::from([V2VisitStep::Node(root)]),
            kind,
            work: 0,
            limit,
            #[cfg(test)]
            copied_bytes: None,
        }
    }

    /// Visits each node once and optionally copies its records in tree order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeV2Signature`] for undefined or repeated node addresses,
    /// [`FormatError::AddressTooLarge`] for addresses outside the file's offset width, and
    /// [`FormatError::ResourceLimit`] when the work allowance is exceeded. Propagates errors from
    /// `read_node` and node parsing.
    pub(super) fn walk<B: AsRef<[u8]>>(
        mut self,
        mut read_node: impl FnMut(BTreeV2Node<'layout>) -> Result<B, FormatError>,
    ) -> Result<Vec<BTreeV2Record>, FormatError> {
        let kind = self.kind;
        let mut records = Vec::new();
        while let Some(step) = self.next()? {
            match step {
                V2VisitStep::Node(node) => {
                    let bytes = read_node(node)?;
                    let view = node.parse(bytes.as_ref())?;
                    if view.child(0)?.is_none() {
                        if matches!(kind, V2WalkKind::Records) {
                            records.reserve(view.record_count());
                            for index in 0..view.record_count() {
                                records.push(self.copy_record(view.record(index)?));
                            }
                        }
                    } else {
                        self.push(&view)?;
                    }
                }
                V2VisitStep::Record(record) => records.push(record),
            }
        }
        Ok(records)
    }

    /// Enables record-byte counting through `counter`.
    #[cfg(test)]
    pub(super) fn with_copy_counter(mut self, counter: Rc<Cell<usize>>) -> Self {
        self.copied_bytes = Some(counter);
        self
    }

    fn copy_record(&self, bytes: &[u8]) -> BTreeV2Record {
        #[cfg(test)]
        if let Some(counter) = &self.copied_bytes {
            counter.set(counter.get().saturating_add(bytes.len()));
        }
        BTreeV2Record {
            data: bytes.to_vec(),
        }
    }

    fn next(&mut self) -> Result<Option<V2VisitStep<'layout>>, FormatError> {
        let Some(step) = self.pending.pop() else {
            return Ok(None);
        };
        if let V2VisitStep::Node(node) = step {
            let address = node.address();
            if address.get() > StoredAddress::undefined(self.width).get() {
                return Err(FormatError::AddressTooLarge {
                    address: address.get(),
                    offset_size: self.width.get(),
                });
            }
            if address.is_undefined(self.width) || !self.visited.insert(address) {
                return Err(FormatError::InvalidBTreeV2Signature);
            }
            self.charge()?;
        }
        Ok(Some(step))
    }

    fn push(&mut self, view: &BTreeV2NodeView<'layout, '_>) -> Result<(), FormatError> {
        for index in (0..=view.record_count()).rev() {
            if index < view.record_count() && matches!(self.kind, V2WalkKind::Records) {
                let record = self.copy_record(view.record(index)?);
                self.pending.push(V2VisitStep::Record(record));
            }
            if let Some(child) = view.child(index)? {
                self.charge()?;
                self.pending.push(V2VisitStep::Node(child));
            }
        }
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

/// A pending node acquisition or a record copied from an internal node.
enum V2VisitStep<'layout> {
    Node(BTreeV2Node<'layout>),
    Record(BTreeV2Record),
}

/// Whether a full walk collects records or only acquires and parses nodes.
#[derive(Clone, Copy)]
pub(super) enum V2WalkKind {
    /// Copies every record in tree order.
    Records,
    /// Acquires and parses nodes without copying records.
    Storage,
}

/// The combined node-visit and child-edge allowance for one full walk.
const DEFAULT_WORK_LIMIT: u64 = 16_777_216;
