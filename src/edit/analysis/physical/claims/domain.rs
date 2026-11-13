//! Evidence that the phase root reaches every supported allocated object header in the source.
//!
//! Valid allocation boundaries are required. Source freshness, complete heap-member
//! ownership, and permission to reuse storage require separate evidence.

use std::sync::Arc;

use hdf5_pure_core::PhysicalAddress;

use super::coverage::content::ResidualOutcome;
use super::{ClaimOrigin, ClaimProblem, PhysicalClaims};
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::{AnalysisPhase, ReconstructionSource};
use crate::error::Error;
use crate::error::FormatError;
use crate::object_graph::ObjectId;

pub(in crate::edit) mod residual;

impl AnalysisPhase<'_> {
    /// Qualifies the original root's reachable objects against the source's physical storage.
    ///
    /// Requires settled allocation publication, complete cached claims with matching allocation
    /// origins, compatible ranges within the recorded bounds, and entirely zero residuals.
    /// Retains independent storage, reference, and residual observations on unavailability.
    /// The caller establishes source consistency for this phase.
    ///
    /// # Errors
    ///
    /// Returns errors from [`Self::physical_claims`], [`Self::physical_coverage`], and
    /// [`Self::residual_content`]. Returns [`Error::Format`] with [`FormatError::Internal`] if
    /// a required cache is absent.
    pub(in crate::edit) fn supported_owner_domain(
        &mut self,
        source: &ReconstructionSource,
    ) -> Result<OwnerDomain, Error> {
        if let Some(reason) = self.domain_restriction(source, CollectionFraming::Complete)? {
            return Ok(OwnerDomain::Unavailable(reason));
        }
        let root = ObjectId::resolve(
            self.engine.addresses(),
            self.engine.superblock().root_group_address,
        )?;
        let session = self.engine.context.allocations.session();
        let claims = self.physical_claims(source)?;
        Ok(OwnerDomain::Available(SupportedOwnerDomain {
            source: ReconstructionSource {
                identity: Arc::clone(&source.identity),
                object: source.object(),
            },
            session,
            root,
            physical_size: claims.physical_size(),
            logical_eoa: claims.logical_eoa(),
        }))
    }

    /// Returns the first heap-domain restriction with header-backed collection allocations.
    ///
    /// Accepts a damaged collection's cached header range only when it matches the recorded claim.
    /// Checks all remaining source, reference, publication, overlap, bounds, and residual
    /// prerequisites. Complete member framing and retirement permission are checked separately.
    ///
    /// # Errors
    ///
    /// Returns the query and cache errors described by [`Self::supported_owner_domain`].
    pub(in crate::edit) fn heap_domain_restriction(
        &mut self,
        source: &ReconstructionSource,
    ) -> Result<Option<DomainUnavailable>, Error> {
        self.domain_restriction(source, CollectionFraming::Envelope)
    }

    /// Returns the first unmet source prerequisite under the selected collection-framing policy.
    ///
    /// # Errors
    ///
    /// Returns errors from [`Self::physical_claims`], [`Self::physical_coverage`], and
    /// [`Self::residual_content`], or [`FormatError::Internal`] if a required cache is absent.
    fn domain_restriction(
        &mut self,
        source: &ReconstructionSource,
        framing: CollectionFraming,
    ) -> Result<Option<DomainUnavailable>, Error> {
        self.physical_claims(source)?;
        if let Err(error) = self.engine.context.allocations.before_reference() {
            return Ok(Some(DomainUnavailable::Publication(error)));
        }
        let Ok(original) = ObjectId::resolve(
            self.engine.addresses(),
            self.engine.superblock().root_group_address,
        ) else {
            return Ok(Some(DomainUnavailable::Root));
        };
        let claims = self
            .claims
            .as_ref()
            .ok_or_else(|| FormatError::Internal("physical claims were not retained".into()))?;
        if claims.root() != Some(original) {
            return Ok(Some(DomainUnavailable::Root));
        }
        if let Some(index) = claims
            .problems()
            .iter()
            .find(|problem| {
                !matches!(
                    (framing, problem),
                    (CollectionFraming::Envelope, ClaimProblem::Collection { address, .. })
                        if self.collection_envelopes.contains_key(address)
                )
            })
            .map(DomainUnavailable::from)
        {
            return Ok(Some(index));
        }
        if let Some(index) = self.unavailable_exclusion(claims, framing)? {
            return Ok(Some(DomainUnavailable::Authority(index)));
        }
        let coverage = self.physical_coverage(source)?;
        if let Some(pair) = coverage.conflict() {
            return Ok(Some(DomainUnavailable::Conflict(pair)));
        }
        if let Some(&index) = coverage.beyond_source().first() {
            return Ok(Some(DomainUnavailable::BeyondSource(index)));
        }
        if let Some(&index) = coverage.beyond_eoa().first() {
            return Ok(Some(DomainUnavailable::BeyondEoa(index)));
        }
        let claims = self.physical_claims(source)?;
        if claims.logical_eoa() > claims.physical_size() {
            return Ok(Some(DomainUnavailable::MissingSource));
        }
        let contents = self.residual_content(source)?;
        if let Some(index) = contents
            .intervals()
            .iter()
            .position(|interval| !matches!(interval.outcome(), ResidualOutcome::AllZero))
        {
            return Ok(Some(DomainUnavailable::Residual(index)));
        }
        Ok(None)
    }

    /// Returns the index of the first claim unsupported by its cached allocation observation.
    ///
    /// Requires each object's claims to be contiguous and ordered as its cached allocations.
    /// Checks heap claims against complete member framing or header ranges selected by `framing`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the physical-storage cache is absent.
    fn unavailable_exclusion(
        &self,
        claims: &PhysicalClaims,
        framing: CollectionFraming,
    ) -> Result<Option<usize>, FormatError> {
        let physical = self
            .physical
            .as_ref()
            .ok_or_else(|| FormatError::Internal("physical storage was not retained".into()))?;
        let mut owner = None;
        let mut allocation_index = 0;
        Ok(claims
            .claims()
            .iter()
            .enumerate()
            .find_map(|(index, claim)| {
                let range = claim.range();
                let authoritative = match claim.origin() {
                    ClaimOrigin::File { index, kind } => physical
                        .file()
                        .get(index)
                        .is_some_and(|entry| entry.range == range && entry.kind == kind),
                    ClaimOrigin::MetadataCapacity(index) => physical
                        .metadata_capacity()
                        .get(index)
                        .is_some_and(|&capacity| capacity == range),
                    ClaimOrigin::NonLive { index, kind } => physical
                        .non_live()
                        .get(index)
                        .is_some_and(|entry| entry.range() == range && entry.kind() == kind),
                    ClaimOrigin::Object { object, kind } => {
                        if owner != Some(object) {
                            owner = Some(object);
                            allocation_index = 0;
                        }
                        let matches = self.ownership.get(&object).is_some_and(|ownership| {
                            ownership.is_complete()
                                && ownership.allocations().get(allocation_index).is_some_and(
                                    |allocation| {
                                        allocation.range() == range && allocation.kind() == kind
                                    },
                                )
                        });
                        allocation_index += 1;
                        matches
                    }
                    ClaimOrigin::Heap(address) => {
                        self.allocations
                            .get(&address)
                            .and_then(|contents| contents.as_ref().ok())
                            .is_some_and(|contents| contents.range() == range)
                            || matches!(framing, CollectionFraming::Envelope)
                                && self.collection_envelopes.get(&address) == Some(&range)
                    }
                };
                (!authoritative).then_some(index)
            }))
    }
}

/// The result of qualifying a source phase's supported allocated object-header domain.
pub(in crate::edit) enum OwnerDomain {
    /// Evidence associated with the original root and allocation session.
    Available(SupportedOwnerDomain),
    /// The first unmet prerequisite, with independent observations retained in the phase.
    Unavailable(DomainUnavailable),
}

/// Evidence that every supported allocated object header is reachable from the original root.
///
/// Valid allocation boundaries are required. Retains the source phase, allocation session, root,
/// physical size, and recorded end address for association checks. Source freshness, complete
/// heap-member ownership, and storage reuse permission require separate evidence.
pub(in crate::edit) struct SupportedOwnerDomain {
    source: ReconstructionSource,
    session: AllocationSession,
    root: ObjectId,
    physical_size: u64,
    logical_eoa: u64,
}

impl SupportedOwnerDomain {
    /// Returns whether the allocation session, source phase, root, and recorded bounds agree.
    ///
    /// The caller establishes freshness after writes, address reuse, or allocator restoration.
    pub(in crate::edit) fn matches(&self, engine: &WriteEngine, claims: &PhysicalClaims) -> bool {
        self.session.same(&engine.context.allocations.session())
            && claims.matches_phase(&self.source)
            && claims.root() == Some(self.root)
            && claims.physical_size() == self.physical_size
            && claims.logical_eoa() == self.logical_eoa
    }
}

/// An unmet prerequisite for a supported source object-header domain.
///
/// Claim indices refer to [`PhysicalClaims::claims`]. File-problem indices refer to
/// [`super::super::PhysicalStorage::problems`], and residual indices refer to
/// [`super::coverage::content::ResidualContents::intervals`].
#[derive(Debug)]
pub(in crate::edit) enum DomainUnavailable {
    /// A claim whose range or origin differs from its cached allocation observation.
    Authority(usize),
    /// A claim beyond the recorded end address.
    BeyondEoa(usize),
    /// A claim beyond the source image length.
    BeyondSource(usize),
    /// A referenced global heap collection with incomplete framing at this physical address.
    Collection(PhysicalAddress),
    /// The two claim indices of an incompatible overlap.
    Conflict([usize; 2]),
    /// The index of a file-storage problem.
    File(usize),
    /// A recorded end address beyond the source image length.
    MissingSource,
    /// An object with incomplete local allocation discovery.
    ObjectStorage(ObjectId),
    /// An allocation-publication failure that prevents qualification.
    Publication(Error),
    /// An object with incomplete reference interpretation.
    References(ObjectId),
    /// An interval with a nonzero witness or incomplete content examination.
    Residual(usize),
    /// An unresolved original root or a different phase root.
    Root,
    /// Unavailable hard-link topology for the phase root.
    Topology,
}

impl From<&ClaimProblem> for DomainUnavailable {
    fn from(problem: &ClaimProblem) -> Self {
        match problem {
            ClaimProblem::Collection { address, .. } => Self::Collection(*address),
            ClaimProblem::File(index) => Self::File(*index),
            ClaimProblem::ObjectStorage(object) => Self::ObjectStorage(*object),
            ClaimProblem::References(object) => Self::References(*object),
            ClaimProblem::Topology => Self::Topology,
        }
    }
}

/// The required allocation observation for a global heap claim.
#[derive(Clone, Copy)]
enum CollectionFraming {
    /// Complete parent and member framing.
    Complete,
    /// Complete framing or a cached allocation range from a parsed supported header.
    Envelope,
}

#[cfg(test)]
mod tests;
