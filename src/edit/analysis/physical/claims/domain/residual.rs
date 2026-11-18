//! Bounded interpretation of residual compact groups for heap-parent qualification.

use core::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::{GroupInfo, LinkInfoMessage, MetadataSource};

use super::{DomainUnavailable, OwnerDomain};
use crate::access_mode::AccessMode;
use crate::convert::Narrow;
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationSession;
use crate::edit::analysis::counts::OrdinaryCountSubject;
use crate::edit::analysis::physical::claims::PhysicalClaims;
use crate::edit::analysis::physical::claims::coverage::content::ResidualOutcome;
use crate::edit::analysis::{AnalysisPhase, ObjectLinks, ObjectReferences, ReconstructionSource};
use crate::error::{Error, FormatError};
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_type::MessageType;
use crate::object_graph::{HardLink, ObjectId};
use crate::object_header::{MessageFilter, ObjectHeader};

impl AnalysisPhase<'_> {
    /// Returns the cached interpretation of residual compact groups, if available.
    ///
    /// Requires the supported owner domain's prerequisites except for zero residual content.
    /// Caches unavailable interpretations and checks the associations of successful ones.
    /// The caller establishes source freshness separately.
    ///
    /// # Errors
    ///
    /// Propagates the errors from [`Self::supported_owner_domain`], or returns
    /// [`FormatError::Internal`] for absent claims or incompatible cached evidence.
    pub(in crate::edit) fn residual_group_domain(
        &mut self,
        source: &ReconstructionSource,
    ) -> Result<Arc<Option<ResidualGroups>>, Error> {
        let cached = self.physical_claims(source)?.residual_groups.clone();
        let result = match cached {
            Some(cached) => cached,
            None => {
                let groups = if matches!(
                    self.supported_owner_domain(source)?,
                    OwnerDomain::Unavailable(DomainUnavailable::Residual(_))
                ) {
                    self.inspect_residual_groups(source)
                } else {
                    None
                };
                let result = Arc::new(groups);
                self.claims
                    .as_mut()
                    .ok_or_else(|| {
                        FormatError::Internal("physical claims were not retained".into())
                    })?
                    .residual_groups = Some(Arc::clone(&result));
                result
            }
        };
        if let Some(groups) = result.as_ref() {
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

    /// Parses nonzero residual intervals and follows their original descendants.
    fn inspect_residual_groups(&mut self, source: &ReconstructionSource) -> Option<ResidualGroups> {
        let intervals = self
            .residual_content(source)
            .ok()?
            .intervals()
            .iter()
            .filter_map(|interval| match interval.outcome() {
                ResidualOutcome::AllZero => None,
                ResidualOutcome::NonZero { .. } => Some(Some(interval.range())),
                ResidualOutcome::ReadFailed(_) => Some(None),
            })
            .collect::<Option<Vec<_>>>()?;
        let mut remaining = MAX_RESIDUAL_BYTES;
        let mut groups = BTreeMap::new();
        let parse_budget = Cell::new(MAX_METADATA_BYTES);
        for range in intervals {
            let length = range.length().to_usize().ok()?;
            remaining = remaining.checked_sub(length)?;
            let bytes = self
                .engine
                .image()
                .read_exact_at(range.start().get(), length)
                .ok()?;
            let metadata = ResidualSource {
                region: FileRegion::new(range, &bytes).ok()?,
                remaining: &parse_budget,
            };
            let mut cursor = 0;
            while let Some(offset) = bytes
                .get(cursor..)
                .and_then(|bytes| bytes.iter().position(|&byte| byte != 0))
            {
                cursor += offset;
                if groups.len() == MAX_RESIDUAL_GROUPS {
                    return None;
                }
                let address = range.start().checked_add(cursor.to_u64()).ok()?;
                let (allocation, links) = metadata.parse_group(self.engine, address)?;
                groups.insert(ObjectId::new(address), links);
                cursor = (allocation.end().get() - range.start().get())
                    .to_usize()
                    .ok()?;
            }
        }
        let topology = self.topology()?;
        let roots = groups.keys().copied().collect::<BTreeSet<_>>();
        let mut objects = roots.clone();
        let mut pending = Vec::new();
        // Every parsed group contributes its links, so only original targets need traversal.
        for group in groups.values() {
            for link in &group.links {
                if topology.indices.contains_key(&link.target) {
                    pending.push(link.target);
                } else if !roots.contains(&link.target) {
                    return None;
                }
            }
        }
        while let Some(object) = pending.pop() {
            if objects.insert(object) {
                pending.extend(topology.children(object));
            }
        }
        let claims = self.physical_claims(source).ok()?;
        let root = claims.root()?;
        let physical_size = claims.physical_size();
        let logical_eoa = claims.logical_eoa();
        let source = self.reconstruction_source(self.engine, root).ok()?;
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
        Some(ResidualGroups {
            source,
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
/// group-information records. Its hard links target objects reachable from the original root
/// or the start of another parsed residual group, including the group itself.
/// Retains the source, allocation session, and bounds for association checks.
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
            && *physical_size == engine.image().len()
            && *logical_eoa == engine.superblock().eof_address
    }
}

/// A metadata view restricted to one physical residual interval and a shared read budget.
struct ResidualSource<'a> {
    region: FileRegion<'a>,
    remaining: &'a Cell<usize>,
}

impl ResidualSource<'_> {
    /// Reads one compact group with only hard links and a single header chunk.
    fn parse_group(
        &self,
        engine: &WriteEngine,
        address: PhysicalAddress,
    ) -> Option<(FileRange, ObjectLinks)> {
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
        .ok()?;
        let [allocation] = allocations.as_slice() else {
            return None;
        };
        if header.version != 2
            || allocation.start() != address
            || allocation.length() == 0
            || !self.region.range().contains(*allocation)
        {
            return None;
        }
        let mut links = Vec::new();
        let mut group = false;
        for message in &header.messages {
            if message.flags.is_shared() {
                return None;
            }
            match message.msg_type {
                MessageType::LINK => {
                    let link = LinkMessage::parse_with_offset_width(
                        &message.data,
                        engine.widths().offsets,
                    )
                    .ok()?;
                    let LinkTarget::Hard {
                        object_header_address,
                    } = link.link_target
                    else {
                        return None;
                    };
                    links.push(HardLink {
                        label: link.name,
                        target: ObjectId::resolve(engine.addresses(), object_header_address)
                            .ok()?,
                    });
                }
                MessageType::LINK_INFO => {
                    let info = LinkInfoMessage::parse_with_offset_width(
                        &message.data,
                        engine.widths().offsets,
                    )
                    .ok()?;
                    if info.fractal_heap_address.is_some()
                        || info.btree_name_index_address.is_some()
                        || info
                            .btree_creation_order_address
                            .is_some_and(|address| address.is_some())
                    {
                        return None;
                    }
                    group = true;
                }
                MessageType::GROUP_INFO => {
                    GroupInfo::parse(&message.data).ok()?;
                    group = true;
                }
                _ => return None,
            }
        }
        group.then(|| {
            (
                *allocation,
                ObjectLinks {
                    links,
                    topology_error: None,
                    count_subject: OrdinaryCountSubject::from_header(&header),
                },
            )
        })
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
mod tests;
