//! Physical placement declarations and recorded object locations.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::iter;

use super::PlacementError;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::commit::plan::Node;
use crate::edit::commit::publish::AppliedTree;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

/// Physical declarations and recorded object placements.
///
/// A successful check establishes agreement with the applied relocation map and root.
/// Source phase, reference counts, predecessor qualification, and publication permission
/// require separate evidence.
pub(in crate::edit::commit) struct PlacementCorrespondence {
    expected: BTreeMap<ObjectPathBuf, Option<ObjectId>>,
    placed: BTreeMap<ObjectPathBuf, ObjectId>,
    error: Option<PlacementError>,
}

impl PlacementCorrespondence {
    /// Iterates each group declaration before that group's relocated datasets.
    ///
    /// Preserves group key order and each group's stored write order. Borrows group paths
    /// and original sources, and joins paths for relocated datasets.
    pub(in crate::edit::commit) fn declarations(
        groups: &BTreeMap<ObjectPathBuf, Node>,
    ) -> impl Iterator<Item = (Cow<'_, ObjectPathBuf>, Option<&ReconstructionSource>)> {
        groups.iter().flat_map(|(path, node)| {
            iter::once((
                Cow::Borrowed(path),
                node.prepared.replaces.as_ref().map(|old| old.source()),
            ))
            .chain(
                node.prepared
                    .writes
                    .iter()
                    .map(move |write| (Cow::Owned(path.join(&write.link)), Some(&write.source))),
            )
        })
    }

    pub(in crate::edit::commit) fn new() -> Self {
        Self {
            expected: BTreeMap::new(),
            placed: BTreeMap::new(),
            error: None,
        }
    }

    /// Sets the expected predecessor at `path`, replacing any earlier declaration.
    ///
    /// A caller passes `None` as the predecessor for a newly created object.
    pub(in crate::edit::commit) fn declare(
        &mut self,
        path: ObjectPathBuf,
        source: Option<ObjectId>,
    ) {
        self.expected.insert(path, source);
    }

    /// Records one successful placement and retains the first event mismatch.
    ///
    /// Checks the path and optional predecessor against the prepared inputs. A newly created
    /// group has no predecessor. Repeated placements are mismatches even at the same address.
    pub(in crate::edit::commit) fn record(
        &mut self,
        path: &ObjectPathBuf,
        source: Option<ObjectId>,
        placed: ObjectId,
    ) {
        let Self {
            expected,
            placed: placements,
            error: placement_error,
        } = self;
        let error = match expected.get(path) {
            None => Some(PlacementError::Unexpected(path.clone())),
            Some(expected) if expected != &source => Some(PlacementError::Source(path.clone())),
            Some(_) if placements.contains_key(path) => {
                Some(PlacementError::Repeated(path.clone()))
            }
            Some(_) => None,
        };
        if let Some(error) = error {
            placement_error.get_or_insert(error);
        } else {
            placements.insert(path.clone(), placed);
        }
    }

    /// Returns the original-to-placed map after checking the applied physical result.
    ///
    /// Uses `originals` independently of the declarations. Omits paths without a recorded
    /// placement. For repeated original identities, uses the last placed entry in iteration
    /// order.
    ///
    /// # Errors
    ///
    /// Returns the recorded mismatch first, then [`PlacementError::Missing`] for the first
    /// unplaced path in key order, [`PlacementError::Relocations`] for a different relocation
    /// map, or [`PlacementError::Root`] for a different root address.
    pub(in crate::edit::commit) fn check<'a>(
        &self,
        tree: &AppliedTree,
        originals: impl IntoIterator<Item = (&'a ObjectPathBuf, ObjectId)>,
    ) -> Result<BTreeMap<ObjectId, ObjectId>, PlacementError> {
        let Self {
            expected,
            placed,
            error,
        } = self;
        if let Some(error) = error {
            return Err(error.clone());
        }
        for path in expected.keys() {
            if !placed.contains_key(path) {
                return Err(PlacementError::Missing(path.clone()));
            }
        }
        let replacements: BTreeMap<_, _> = originals
            .into_iter()
            .filter_map(|(path, original)| placed.get(path).map(|&placed| (original, placed)))
            .collect();
        if replacements != tree.relocations {
            return Err(PlacementError::Relocations);
        }
        if placed
            .get(&ObjectPathBuf::root())
            .map(|root| root.header_address())
            != Some(tree.retirement.root())
        {
            return Err(PlacementError::Root);
        }
        Ok(replacements)
    }

    /// Replaces any recorded mismatch with a duplicate declaration error for `path`.
    pub(in crate::edit::commit::plan::inventory::dispositions) fn repeated(
        &mut self,
        path: ObjectPathBuf,
    ) {
        self.error = Some(PlacementError::Repeated(path));
    }
}
