//! Prospective committed-datatype user counts interpreted in one analysis phase.
//!
//! The caller supplies same-file copy models and accounts for every planned emission. Evidence
//! that the copy was captured from the same unchanged file, original-user minima, allocation
//! ownership, and count-write permissions remain separate.

use core::num::NonZeroU64;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::sync::Arc;

use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::analysis::deletions::TopologyIdentity;
use crate::edit::copy::CopyTree;
use crate::edit::copy::datatype_users::CopiedDatatypeUsers;
use crate::error::{Error, FormatError};
use crate::object_graph::ObjectId;

impl<'source> AnalysisPhase<'source> {
    /// Interprets a copy's committed-datatype users with this engine's field widths.
    ///
    /// Returns `None` for an incomplete inventory. A complete empty inventory has no phase
    /// association. For a nonempty inventory, the result borrows `tree` and contains this engine,
    /// root, and lazy phase identity. Source reads and topology discovery are deferred.
    ///
    /// The caller passes a model captured from this engine's file with matching field widths and
    /// preserves the source bytes on which the model depends. The engine, root, and phase identity
    /// describe destination interpretation. Copy-source freshness remains a separate requirement.
    ///
    /// # Errors
    ///
    /// Returns errors from [`CopyTree::datatype_users`] or checked root-address resolution.
    pub(in crate::edit) fn copy_users<'copy>(
        &mut self,
        tree: &'copy CopyTree,
    ) -> Result<Option<PhaseCopyUsers<'copy, 'source>>, Error> {
        let Some(users) = tree.datatype_users(self.engine.widths())? else {
            return Ok(None);
        };
        let source = users
            .users()
            .next()
            .map(|_| CopyUsersSource::new(self))
            .transpose()?;
        Ok(Some(PhaseCopyUsers { source, users }))
    }

    /// Counts users across the supplied actual copy emissions in this phase.
    ///
    /// The caller includes every emission once. Repeating a bound inventory counts another
    /// emission of that copy. These multiplicities are separate from distinct original users.
    /// Returns `None` at the first incomplete inventory and drops the accumulated prefix.
    /// Empty inventories are neutral, including inventories interpreted in another phase.
    ///
    /// Retains one ordered map of physical targets. Each consumed inventory entry requires a
    /// checked address conversion and an ordered-map lookup. Inventory parsing and copy-model
    /// storage have separate costs. Processing stops at the first incomplete inventory or error.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`CopyUsersUnion::add`]. The accumulated prefix is dropped on error.
    pub(in crate::edit) fn added_copy_users<'copy>(
        &self,
        copies: impl IntoIterator<Item = Option<&'copy PhaseCopyUsers<'copy, 'source>>>,
    ) -> Result<Option<AddedCopyUsers<'source>>, CopyUsersError>
    where
        'source: 'copy,
    {
        let mut union = CopyUsersUnion::new();
        for copy in copies {
            let Some(next) = union.add(self, copy)? else {
                return Ok(None);
            };
            union = next;
        }
        Ok(Some(union.finish()))
    }

    fn copy_user_root(&self) -> Result<ObjectId, FormatError> {
        self.root.map_or_else(
            || {
                ObjectId::resolve(
                    self.engine.addresses(),
                    self.engine.superblock().root_group_address,
                )
            },
            Ok,
        )
    }
}

/// An incremental union of complete inventories with one nonempty phase association.
///
/// The caller can release each copy inventory after adding it. Only physical target counts and
/// the optional phase association are retained.
pub(in crate::edit) struct CopyUsersUnion<'source>(AddedCopyUsers<'source>);

impl<'source> CopyUsersUnion<'source> {
    /// Starts an empty union with deferred phase association and lookup storage.
    pub(in crate::edit) fn new() -> Self {
        Self(AddedCopyUsers {
            source: None,
            users: BTreeMap::new(),
        })
    }

    /// Adds one actual emission and consumes the accumulated prefix.
    ///
    /// Requires matching engine, root, and phase identities for nonempty inventories. Returns
    /// `None` for an incomplete inventory. The prefix is dropped on incompleteness or error.
    /// For empty inventories, the union's association is unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`CopyUsersError::Engine`], [`CopyUsersError::Root`], or [`CopyUsersError::Phase`]
    /// for an incompatible association, [`CopyUsersError::Address`] for failed root or target
    /// resolution, or [`CopyUsersError::Overflow`] if a target's accumulated count exceeds `u64`.
    pub(in crate::edit) fn add(
        mut self,
        phase: &AnalysisPhase<'source>,
        copy: Option<&PhaseCopyUsers<'_, 'source>>,
    ) -> Result<Option<Self>, CopyUsersError> {
        let Some(PhaseCopyUsers {
            source: copy_source,
            users: copied,
        }) = copy
        else {
            return Ok(None);
        };
        let Some(copy_source) = copy_source else {
            return Ok(Some(self));
        };
        let Self(added) = &mut self;
        let AddedCopyUsers { source, users } = added;
        copy_source.require_phase(phase)?;
        if let Some(source) = source {
            copy_source.require(source)?;
        } else {
            *source = Some(copy_source.clone());
        }
        for (address, count) in copied.users() {
            let object = ObjectId::resolve(phase.engine.addresses(), address)
                .map_err(CopyUsersError::Address)?;
            add_users(users, object, count)?;
        }
        Ok(Some(self))
    }

    /// Returns the complete counts accumulated from the supplied emissions.
    pub(in crate::edit) fn finish(self) -> AddedCopyUsers<'source> {
        let Self(added) = self;
        added
    }
}

/// A complete borrowed copy inventory with an optional destination-phase association.
///
/// For nonempty inventories, the result contains the engine, root, and phase identity. The
/// inventory immutably borrows its copy model. The caller establishes that the copy was captured
/// from the same unchanged file.
pub(in crate::edit) struct PhaseCopyUsers<'copy, 'source> {
    source: Option<CopyUsersSource<'source>>,
    users: CopiedDatatypeUsers<'copy>,
}

/// Checked prospective multiplicities for physical targets in one nonempty analysis phase.
///
/// An empty result has no phase association. The caller establishes completeness of the supplied
/// emission set. Original-user evidence, allocation ownership, and publication remain separate.
pub(in crate::edit) struct AddedCopyUsers<'source> {
    source: Option<CopyUsersSource<'source>>,
    users: BTreeMap<ObjectId, NonZeroU64>,
}

impl AddedCopyUsers<'_> {
    /// Returns the associated physical root, or `None` for empty counts.
    pub(in crate::edit) fn root(&self) -> Option<ObjectId> {
        self.source.as_ref().map(|source| source.root)
    }

    /// Returns positive multiplicities in physical object-header address order.
    pub(in crate::edit) fn users(&self) -> impl Iterator<Item = (ObjectId, NonZeroU64)> + '_ {
        self.users.iter().map(|(&object, &count)| (object, count))
    }
}

/// An engine, root, and lazy phase identity used for destination interpretation.
#[derive(Clone)]
struct CopyUsersSource<'source> {
    engine: &'source WriteEngine,
    identity: Arc<TopologyIdentity>,
    root: ObjectId,
}

impl<'source> CopyUsersSource<'source> {
    fn new(phase: &mut AnalysisPhase<'source>) -> Result<Self, FormatError> {
        let engine = phase.engine;
        let root = phase.copy_user_root()?;
        let ReconstructionSource { identity, object } =
            phase.reconstruction_source(engine, root)?;
        Ok(Self {
            engine,
            identity,
            root: object,
        })
    }

    fn require_phase(&self, phase: &AnalysisPhase<'_>) -> Result<(), CopyUsersError> {
        let Self {
            engine,
            identity,
            root,
        } = self;
        if !core::ptr::eq(*engine, phase.engine) {
            return Err(CopyUsersError::Engine);
        }
        let expected = phase.copy_user_root().map_err(CopyUsersError::Address)?;
        if *root != expected {
            return Err(CopyUsersError::Root {
                expected,
                actual: *root,
            });
        }
        if !phase
            .identity
            .as_ref()
            .is_some_and(|other| Arc::ptr_eq(identity, other))
        {
            return Err(CopyUsersError::Phase);
        }
        Ok(())
    }

    fn require(&self, expected: &Self) -> Result<(), CopyUsersError> {
        let Self {
            engine,
            identity,
            root,
        } = self;
        let Self {
            engine: expected_engine,
            identity: expected_identity,
            root: expected_root,
        } = expected;
        if !core::ptr::eq(*engine, *expected_engine) {
            return Err(CopyUsersError::Engine);
        }
        if root != expected_root {
            return Err(CopyUsersError::Root {
                expected: *expected_root,
                actual: *root,
            });
        }
        if !Arc::ptr_eq(identity, expected_identity) {
            return Err(CopyUsersError::Phase);
        }
        Ok(())
    }
}

fn add_users(
    users: &mut BTreeMap<ObjectId, NonZeroU64>,
    object: ObjectId,
    count: NonZeroU64,
) -> Result<(), CopyUsersError> {
    match users.entry(object) {
        Entry::Occupied(mut entry) => {
            let current = *entry.get();
            let next = current
                .checked_add(count.get())
                .ok_or(CopyUsersError::Overflow {
                    object,
                    current,
                    added: count,
                })?;
            entry.insert(next);
        }
        Entry::Vacant(entry) => {
            entry.insert(count);
        }
    }
    Ok(())
}

/// An incompatible destination phase, failed address resolution, or count overflow.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit) enum CopyUsersError {
    /// A checked root or stored-target address conversion failed.
    Address(FormatError),
    /// A nonempty inventory belongs to another engine.
    Engine,
    /// An accumulated target count exceeds `u64`.
    Overflow {
        object: ObjectId,
        current: NonZeroU64,
        added: NonZeroU64,
    },
    /// A nonempty inventory belongs to another analysis phase.
    Phase,
    /// A nonempty inventory is associated with another root.
    Root {
        expected: ObjectId,
        actual: ObjectId,
    },
}

#[cfg(test)]
mod tests;
