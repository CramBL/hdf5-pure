//! Prospective declared-count additions within the shared original-edge removal traversal.

use core::num::NonZeroU32;

use crate::edit::analysis::counts::CountEvidence;
use crate::edit::analysis::counts::CountEvidenceError;
use crate::edit::analysis::counts::balance::{CountArithmeticError, CountBalance};
use crate::error::FormatError;
use crate::object_graph::{ObjectId, ObjectIndex};

use super::{CompleteTopology, LinkRemovalPlan, RemovedLinks, SourceCountState, SourceRemovalPlan};

impl<'topology, 'source> SourceRemovalPlan<'topology, 'source> {
    /// Plans declared net counts through the shared original-edge removal traversal.
    ///
    /// Calls `query` once per touched object and checks the evidence against the same source,
    /// topology, physical object, and original incoming count. For matching known evidence, calls
    /// `additions` once with that object. Original evidence is checked before the addition lookup.
    /// Returns `None` without topology, before calling either callback.
    ///
    /// An object with positive additions is ineligible for outgoing propagation. Final net counts
    /// are checked against [`u32::MAX`] after the original-edge removals. Consumes the touched
    /// states through one ordered-map collection. The private plan is dropped on error.
    ///
    /// The caller supplies each touched target's complete prospective multiplicity and separately
    /// accounts for additions to untouched or outside targets. Same-file applicability, source
    /// freshness, allocation ownership, retirement, and publication require separate evidence.
    ///
    /// # Errors
    ///
    /// Returns [`AddedCountError::Evidence`] for count-query errors, original contradictions, or
    /// inconsistent removal or evidence associations. Returns [`AddedCountError::Arithmetic`] for
    /// an original-count underflow or a final net count exceeding [`u32::MAX`].
    pub(super) fn with_additions(
        removed: RemovedLinks<'topology, 'source>,
        mut query: impl FnMut(ObjectId) -> Result<Option<CountEvidence<'source>>, CountEvidenceError>,
        mut additions: impl FnMut(ObjectId) -> u64,
    ) -> Result<Option<Self>, AddedCountError> {
        let RemovedLinks {
            topology,
            edges,
            unresolved: _,
        } = removed;
        let Some(topology) = topology else {
            return Ok(None);
        };
        let plan = LinkRemovalPlan::build(
            topology,
            edges,
            |index| DeferredCountState::new(topology, index, &mut query, &mut additions),
            DeferredCountState::remove,
        )?;
        let LinkRemovalPlan {
            topology,
            remaining,
            zero,
            #[cfg(test)]
            work,
        } = plan;
        let remaining = remaining
            .into_iter()
            .map(|(index, state)| state.finish().map(|state| (index, state)))
            .collect::<Result<_, _>>()?;
        Ok(Some(Self {
            plan: LinkRemovalPlan {
                topology,
                remaining,
                zero,
                #[cfg(test)]
                work,
            },
        }))
    }
}

/// A graph remainder and an optional declared balance pending finalization.
struct DeferredCountState<'source> {
    graph: u32,
    declared: Option<DeferredDeclaredCount<'source>>,
}

impl<'source> DeferredCountState<'source> {
    /// Checks original graph and declared evidence before looking up prospective additions.
    ///
    /// The addition callback is called only for matching known count evidence.
    ///
    /// # Errors
    ///
    /// Returns [`AddedCountError::Evidence`] for the errors of [`SourceCountState::new`].
    fn new(
        topology: &CompleteTopology<'source>,
        index: ObjectIndex,
        query: &mut impl FnMut(ObjectId) -> Result<Option<CountEvidence<'source>>, CountEvidenceError>,
        additions: &mut impl FnMut(ObjectId) -> u64,
    ) -> Result<Self, AddedCountError> {
        let SourceCountState {
            evidence,
            graph,
            declared: _,
        } = SourceCountState::new(topology, index, query).map_err(AddedCountError::Evidence)?;
        let declared = evidence.map(|evidence| DeferredDeclaredCount {
            balance: CountBalance::new(
                evidence.bound().original(),
                additions(evidence.bound().object()),
            ),
            evidence,
        });
        Ok(Self { graph, declared })
    }

    /// Decrements the graph and any original-count remainder before testing eligibility.
    ///
    /// Returns `true` for an ordinary subject with zero graph and net counts. On error, both
    /// remainders are unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`AddedCountError::Evidence`] if the graph count would underflow, or
    /// [`AddedCountError::Arithmetic`] if the original declared remainder would underflow.
    fn remove(&mut self) -> Result<bool, AddedCountError> {
        let Self { graph, declared } = self;
        let next_graph = graph.checked_sub(1).ok_or_else(|| {
            FormatError::Internal("removed links exceed their target's incoming count".into())
        })?;
        if let Some(DeferredDeclaredCount { evidence, balance }) = declared {
            balance
                .remove(NonZeroU32::MIN)
                .map_err(|error| AddedCountError::Arithmetic {
                    object: evidence.bound().object(),
                    error,
                })?;
        }
        *graph = next_graph;
        Ok(*graph == 0
            && declared.as_ref().is_some_and(|state| {
                state.balance.is_zero() && matches!(state.evidence, CountEvidence::Ordinary(_))
            }))
    }

    /// Checks the final net count and retains the unchanged original evidence.
    ///
    /// # Errors
    ///
    /// Returns [`AddedCountError::Arithmetic`] if the declared net count exceeds [`u32::MAX`].
    fn finish(self) -> Result<SourceCountState<'source>, AddedCountError> {
        let Self { graph, declared } = self;
        let (evidence, declared) = match declared {
            Some(DeferredDeclaredCount { evidence, balance }) => {
                let remaining =
                    balance
                        .remaining()
                        .map_err(|error| AddedCountError::Arithmetic {
                            object: evidence.bound().object(),
                            error,
                        })?;
                (Some(evidence), Some(remaining))
            }
            None => (None, None),
        };
        Ok(SourceCountState {
            evidence,
            graph,
            declared,
        })
    }
}

/// Matching original-count evidence paired with its deferred prospective arithmetic.
struct DeferredDeclaredCount<'source> {
    evidence: CountEvidence<'source>,
    balance: CountBalance,
}

/// An original-count evidence error or failed net-count arithmetic.
#[derive(Debug, Eq, PartialEq)]
pub(super) enum AddedCountError {
    /// A checked count-arithmetic failure for a physical target.
    Arithmetic {
        /// The physical object-header identity of the affected count.
        object: ObjectId,
        /// The arithmetic failure and its original operands.
        error: CountArithmeticError,
    },
    /// A count-query failure, original contradiction, or inconsistent removal topology.
    Evidence(CountEvidenceError),
}

impl From<FormatError> for AddedCountError {
    /// Classifies a structural planning failure as an evidence error.
    fn from(error: FormatError) -> Self {
        Self::Evidence(error.into())
    }
}
