//! Bounded interpretation of residual compact groups for heap-parent qualification.

use core::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::{GroupInfo, GroupInfoError, LinkInfoMessage, MetadataSource};

use super::DomainUnavailable;
use crate::access_mode::AccessMode;
use crate::convert::Narrow;
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::physical::claims::PhysicalClaims;
use crate::edit::analysis::physical::claims::coverage::content::ResidualOutcome;
use crate::edit::analysis::{AnalysisPhase, ObjectLinks, ObjectReferences, ReconstructionSource};
use crate::error::{Error, FormatError};
use crate::message_type::MessageType;
use crate::object_graph::ObjectId;
use crate::object_header::{MessageFilter, ObjectHeader};

impl AnalysisPhase<'_> {
    /// Returns cached residual-group evidence or the outcome that prevents qualification.
    ///
    /// Requires the heap-domain prerequisites other than zero residual content, accepting
    /// independently observed collection-header allocation ranges after member framing fails.
    /// Retains successful and unavailable interpretations in this phase. Checks successful evidence
    /// against the source phase, allocation session, root, physical size, and recorded end address.
    /// The caller establishes source freshness separately.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::physical_claims`] or [`FormatError::Internal`] for an absent
    /// cache or incompatible retained associations. Records interpretation, read, and limit
    /// failures in the returned [`ResidualUnavailable`].
    pub(in crate::edit) fn residual_group_domain(
        &mut self,
        source: &ReconstructionSource,
    ) -> Result<Arc<Result<ResidualGroups, ResidualUnavailable>>, Error> {
        let cached = self.physical_claims(source)?.residual_groups.clone();
        let result = match cached {
            Some(cached) => cached,
            None => {
                let result = Arc::new(self.inspect_residual_groups(source));
                self.claims
                    .as_mut()
                    .ok_or_else(|| {
                        FormatError::Internal("physical claims were not retained".into())
                    })?
                    .residual_groups = Some(result.clone());
                result
            }
        };
        if let Ok(groups) = result.as_ref() {
            let engine = self.engine;
            if !groups.matches(engine, source, self.physical_claims(source)?) {
                return Err(FormatError::Internal(
                    "residual groups belong to another source phase or session".into(),
                )
                .into());
            }
        }
        Ok(result)
    }

    /// Parses residual groups and retains their original links and reference-free summaries.
    ///
    /// # Errors
    ///
    /// Returns [`ResidualUnavailable`] if domain, content, parsing, link-target, or resource
    /// prerequisites fail.
    fn inspect_residual_groups(
        &mut self,
        source: &ReconstructionSource,
    ) -> Result<ResidualGroups, ResidualUnavailable> {
        match self
            .heap_domain_restriction(source)
            .map_err(ResidualUnavailable::Source)?
        {
            Some(DomainUnavailable::Residual(_)) => {}
            Some(reason) => return Err(ResidualUnavailable::Domain(reason)),
            None => {}
        }
        let intervals = self
            .residual_content(source)
            .map_err(ResidualUnavailable::Source)?
            .intervals()
            .iter()
            .filter_map(|interval| match interval.outcome() {
                ResidualOutcome::AllZero => None,
                ResidualOutcome::NonZero { .. } => Some(Ok(interval.range())),
                ResidualOutcome::ReadFailed(error) => {
                    Some(Err(ResidualUnavailable::Format(error.clone())))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut remaining = MAX_RESIDUAL_BYTES;
        let mut groups = BTreeMap::new();
        let parse_budget = Cell::new(MAX_METADATA_BYTES);
        for range in intervals {
            if range.length() > remaining.to_u64() {
                return Err(ResidualUnavailable::Limit {
                    kind: ResidualLimit::Bytes,
                    required: (MAX_RESIDUAL_BYTES - remaining)
                        .to_u64()
                        .saturating_add(range.length()),
                    maximum: MAX_RESIDUAL_BYTES.to_u64(),
                });
            }
            let length = range
                .length()
                .to_usize()
                .map_err(ResidualUnavailable::Format)?;
            remaining -= length;
            let bytes = self
                .engine
                .image()
                .read_exact_at(range.start().get(), length)
                .map_err(ResidualUnavailable::Format)?;
            let region = FileRegion::new(range, &bytes).map_err(ResidualUnavailable::Format)?;
            let metadata = ResidualSource {
                region,
                remaining: &parse_budget,
            };
            let mut cursor = 0;
            while let Some(offset) = bytes
                .get(cursor..)
                .and_then(|bytes| bytes.iter().position(|&byte| byte != 0))
            {
                cursor += offset;
                if groups.len() == MAX_RESIDUAL_GROUPS {
                    return Err(ResidualUnavailable::Limit {
                        kind: ResidualLimit::Groups,
                        required: (groups.len() + 1).to_u64(),
                        maximum: MAX_RESIDUAL_GROUPS.to_u64(),
                    });
                }
                let address = range
                    .start()
                    .checked_add(cursor.to_u64())
                    .map_err(ResidualUnavailable::Format)?;
                let (allocation, links) = metadata.parse_group(self.engine, address)?;
                groups.insert(ObjectId::new(address), links);
                cursor = (allocation.end().get() - range.start().get())
                    .to_usize()
                    .map_err(ResidualUnavailable::Format)?;
            }
        }
        let known = self
            .topology()
            .ok_or(ResidualUnavailable::Topology)?
            .indices
            .keys()
            .copied()
            .collect::<BTreeSet<_>>();
        if let Some((&object, _)) = groups.iter().find(|(_, links)| {
            links
                .links
                .iter()
                .any(|link| !known.contains(&link.target) && !groups.contains_key(&link.target))
        }) {
            return Err(ResidualUnavailable::Link(object));
        }
        let roots = groups.keys().copied().collect::<BTreeSet<_>>();
        let mut objects = BTreeSet::new();
        let mut pending = roots.iter().copied().collect::<Vec<_>>();
        let topology = self.topology().ok_or(ResidualUnavailable::Topology)?;
        while let Some(object) = pending.pop() {
            if !objects.insert(object) {
                continue;
            }
            if let Some(group) = groups.get(&object) {
                pending.extend(group.links.iter().map(|link| link.target));
            } else {
                pending.extend(topology.children(object));
            }
        }
        for (object, links) in groups {
            self.links.insert(object, links);
            self.references.insert(
                object,
                ObjectReferences {
                    complete: true,
                    ..ObjectReferences::default()
                },
            );
        }
        let claims = self
            .physical_claims(source)
            .map_err(ResidualUnavailable::Source)?;
        let root = claims.root().ok_or(ResidualUnavailable::Topology)?;
        let physical_size = claims.physical_size();
        let logical_eoa = claims.logical_eoa();
        Ok(ResidualGroups {
            source: self
                .reconstruction_source(self.engine, root)
                .map_err(ResidualUnavailable::Format)?,
            session: self.engine.context.allocations.session(),
            physical_size,
            logical_eoa,
            roots,
            objects,
        })
    }
}

/// Complete bounded interpretation of residual zero bytes and compact group headers.
///
/// Each group occupies one header chunk containing only unshared link, link-information, or
/// group-information records. Its hard links target original-root objects or other parsed residual
/// groups. The original source, allocation session, and bounds are retained for association checks.
/// Source freshness and storage reuse require separate evidence.
pub(in crate::edit) struct ResidualGroups {
    source: ReconstructionSource,
    session: AllocationSession,
    physical_size: u64,
    logical_eoa: u64,
    /// Original physical identities of the parsed residual groups.
    roots: BTreeSet<ObjectId>,
    /// Original objects reachable from those groups, including the groups themselves.
    objects: BTreeSet<ObjectId>,
}

impl ResidualGroups {
    /// Borrows the original root's source-phase association.
    pub(in crate::edit) fn source(&self) -> &ReconstructionSource {
        &self.source
    }

    /// Borrows the allocation session retained during interpretation.
    pub(in crate::edit) fn session(&self) -> &AllocationSession {
        &self.session
    }

    /// Borrows original residual-group identities for prospective root translation.
    pub(in crate::edit) fn roots(&self) -> &BTreeSet<ObjectId> {
        &self.roots
    }

    /// Borrows original identities requiring protection from retirement and reuse.
    pub(in crate::edit) fn objects(&self) -> &BTreeSet<ObjectId> {
        &self.objects
    }

    /// Returns whether the retained source phase, session, root, and bounds match.
    fn matches(
        &self,
        engine: &WriteEngine,
        source: &ReconstructionSource,
        claims: &PhysicalClaims,
    ) -> bool {
        let Self {
            source: original,
            session,
            physical_size,
            logical_eoa,
            roots: _,
            objects: _,
        } = self;
        session.same(&engine.context.allocations.session())
            && Arc::ptr_eq(&original.identity, &source.identity)
            && Some(original.object()) == claims.root()
            && *physical_size == claims.physical_size()
            && *logical_eoa == claims.logical_eoa()
    }
}

/// An unmet prerequisite for interpreting residual storage as compact groups and zero bytes.
#[derive(Debug)]
#[expect(
    dead_code,
    reason = "the outcome retains the source and framing observations that withhold qualification"
)]
pub(in crate::edit) enum ResidualUnavailable {
    /// A heap-domain prerequisite other than residual content is unavailable.
    Domain(DomainUnavailable),
    /// A residual read or parse failure, including an exhausted metadata budget.
    Format(FormatError),
    /// A residual group's group-information record is invalid.
    Group {
        object: ObjectId,
        error: GroupInfoError,
    },
    /// The aggregate residual-byte or group-count limit was exceeded.
    Limit {
        kind: ResidualLimit,
        required: u64,
        maximum: u64,
    },
    /// A residual group has a hard-link target outside the known original and residual objects.
    Link(ObjectId),
    /// Source-domain, coverage, or link discovery failed.
    Source(Error),
    /// The original topology or physical-root association is unavailable.
    Topology,
    /// A residual object is outside the supported compact-group subset.
    Unsupported(ObjectId),
}

/// The bounded quantity that prevented complete residual interpretation.
#[derive(Debug)]
pub(in crate::edit) enum ResidualLimit {
    /// Total bytes in residual intervals with nonzero content.
    Bytes,
    /// The number of residual group headers.
    Groups,
}

/// A metadata view restricted to one physical residual interval and a shared read budget.
struct ResidualSource<'a> {
    region: FileRegion<'a>,
    remaining: &'a Cell<usize>,
}

impl ResidualSource<'_> {
    /// Reads a single compact group wholly contained in this residual interval.
    ///
    /// # Errors
    ///
    /// Returns [`ResidualUnavailable::Unsupported`] for multiple chunks, shared or unsupported
    /// records, dense links, invalid local bounds, or undefined targets. Returns
    /// [`ResidualUnavailable::Group`] for invalid group information,
    /// [`ResidualUnavailable::Format`] for header or link-information parsing failures, or
    /// [`ResidualUnavailable::Source`] for link discovery errors.
    fn parse_group(
        &self,
        engine: &WriteEngine,
        address: PhysicalAddress,
    ) -> Result<(FileRange, ObjectLinks), ResidualUnavailable> {
        let mut allocations = Vec::new();
        let header = ObjectHeader::parse_from_source_with_chunk_observer(
            self,
            AccessMode::ReadWrite,
            address,
            engine.widths(),
            engine.addresses().effective_base(),
            MessageFilter::All,
            &mut |chunk| allocations.push(chunk.allocation()),
        )
        .map_err(ResidualUnavailable::Format)?;
        let [allocation] = allocations.as_slice() else {
            return Err(ResidualUnavailable::Unsupported(ObjectId::new(address)));
        };
        if allocation.start() != address
            || allocation.length() == 0
            || !self.region.range().contains(*allocation)
        {
            return Err(ResidualUnavailable::Unsupported(ObjectId::new(address)));
        }
        for message in &header.messages {
            if message.flags.is_shared() {
                return Err(ResidualUnavailable::Unsupported(ObjectId::new(address)));
            }
            match message.msg_type {
                MessageType::LINK => {}
                MessageType::LINK_INFO => {
                    let info = LinkInfoMessage::parse_with_offset_width(
                        &message.data,
                        engine.widths().offsets,
                    )
                    .map_err(ResidualUnavailable::Format)?;
                    if info.fractal_heap_address.is_some()
                        || info.btree_name_index_address.is_some()
                        || info
                            .btree_creation_order_address
                            .is_some_and(|address| address.is_some())
                    {
                        return Err(ResidualUnavailable::Unsupported(ObjectId::new(address)));
                    }
                }
                MessageType::GROUP_INFO => {
                    GroupInfo::parse(&message.data).map_err(|error| {
                        ResidualUnavailable::Group {
                            object: ObjectId::new(address),
                            error,
                        }
                    })?;
                }
                _ => return Err(ResidualUnavailable::Unsupported(ObjectId::new(address))),
            }
        }
        if !header.messages.iter().any(|message| {
            matches!(
                message.msg_type,
                MessageType::LINK_INFO | MessageType::GROUP_INFO
            )
        }) {
            return Err(ResidualUnavailable::Unsupported(ObjectId::new(address)));
        }
        let links = ObjectLinks::read(engine, &header).map_err(ResidualUnavailable::Source)?;
        if links.topology_error.is_some() {
            return Err(ResidualUnavailable::Unsupported(ObjectId::new(address)));
        }
        Ok((*allocation, links))
    }
}

impl MetadataSource for ResidualSource<'_> {
    type Bytes<'a>
        = &'a [u8]
    where
        Self: 'a;

    fn read_metadata(&self, offset: u64, length: usize) -> Result<Self::Bytes<'_>, FormatError> {
        let Self { region, remaining } = self;
        let bytes = region
            .subregion(FileRange::new(
                PhysicalAddress::new(offset),
                length.to_u64(),
            )?)?
            .bytes();
        let available = remaining.get();
        let unused = available
            .checked_sub(length)
            .ok_or(FormatError::ResourceLimit {
                limit: MAX_METADATA_BYTES.to_u64(),
                required: (MAX_METADATA_BYTES - available)
                    .to_u64()
                    .saturating_add(length.to_u64()),
            })?;
        remaining.set(unused);
        Ok(bytes)
    }

    fn len(&self) -> u64 {
        self.region.range().end().get()
    }

    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        bytes.copy_from_slice(self.read_metadata(offset, bytes.len())?);
        Ok(())
    }

    fn read_metadata_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, FormatError> {
        Ok(self.read_metadata(offset, length)?.to_vec())
    }
}

/// Maximum total bytes in residual intervals with nonzero content.
const MAX_RESIDUAL_BYTES: usize = 1_048_576;
/// Maximum total metadata bytes requested while parsing residual groups.
const MAX_METADATA_BYTES: usize = 2_097_152;
/// Maximum number of residual group headers.
const MAX_RESIDUAL_GROUPS: usize = 64;

#[cfg(test)]
pub(in crate::edit) mod tests;
