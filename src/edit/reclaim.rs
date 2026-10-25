//! Collects the storage a commit frees, with the [`FreeClass`] of each span.
//!
//! A commit frees an object's storage when it deletes the object's last hard link, a global heap
//! collection when the objects it deletes are the only ones that refer to the collection, and the
//! storage of a dense attribute set or a chunk index when it replaces one. The walk returns a span
//! only once it accounts for every allocation of the structure the span belongs to, and leaves a
//! structure it cannot account for in full allocated.

use std::collections::{HashMap, HashSet};

use hdf5_pure_core::__private::PhysicalAddress;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;

use crate::access_mode::AccessMode;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::attribute_info::AttributeInfoMessage;
use crate::chunked_read;
use crate::data_layout::{ChunkIndexLayout, DataLayout};
use crate::dataspace::Dataspace;
use crate::error::{Error, FormatError};
use crate::file_space_info::FileSpacePageSize;
use crate::fractal_heap_storage;
use crate::fractal_heap_storage::FractalHeapAllocation;
use crate::fractal_heap_storage::FractalHeapStorageError;
use crate::global_heap;
use crate::group_v1;
use crate::group_v2;
use crate::link_info::LinkInfoMessage;
use crate::link_message::LinkMessage;
use crate::link_message::LinkTarget;
use crate::message_type::MessageType;
use crate::object_header::HeaderMessage;
use crate::object_header::ObjectHeader;
use crate::source::{BaseOffsetSource, Source, SourceMetadata};

use super::GlobalHeapLiveness;
use super::copy::{MAX_COPY_DEPTH, ObjModel};
use super::{MAX_LINK_GRAPH_NODES, OhRegion, SpaceOwnership, WriteEngine};

/// The spans `(address, length)` of the global heap collections that the objects a commit deletes
/// refer to, as [`WriteEngine::deleted_collections`] returns them.
pub(super) struct DeletedCollections(Vec<(u64, u64)>);

struct ChunkedStorageParts {
    data: Vec<(u64, u64)>,
    index: Vec<(u64, u64)>,
    index_layout: ChunkIndexLayout,
}

impl WriteEngine {
    /// Returns the byte ownership this editor can identify for a committed file.
    ///
    /// The editor visits each reachable object once, including objects with several hard
    /// links or cycles. The identified storage includes the userblock, the superblock, its
    /// extension, persisted free-space managers, reusable free extents, and the global heap
    /// collections that reachable objects refer to. Unsupported storage or a traversal limit makes
    /// the analysis incomplete, as does a [`GlobalHeapLiveness`] analysis that is incomplete.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if supported reachable metadata, a variable-length element, or a
    /// global heap collection that a reachable object refers to cannot be parsed. Returns
    /// [`Error::MissingMessage`] if a dataset lacks its datatype or dataspace message. Returns
    /// [`Error::InvalidFractalHeapStorage`] if a heap has inconsistent allocation geometry.
    /// Returns [`Error::InvalidSpaceOwnershipExtent`] if identified extents overlap or exceed
    /// the logical end of file.
    pub(crate) fn space_ownership(&self) -> Result<SpaceOwnership, Error> {
        let physical_size = self.image.len();
        let logical_size = self.context.superblock().eof_address;
        if logical_size > physical_size {
            return Err(FormatError::TruncatedFile {
                expected_eof: logical_size,
                actual_len: physical_size,
            }
            .into());
        }

        let mut live_raw_space = Vec::new();
        let mut live_metadata_space = Vec::new();
        let root = self
            .context
            .addresses()
            .absolute(self.context.superblock().root_group_address)?;
        let mut analysis_complete =
            self.collect_owned_spans(root, &mut live_raw_space, &mut live_metadata_space)?;
        let GlobalHeapLiveness {
            referenced_collections,
            analysis_complete: collections_complete,
        } = self.global_heap_liveness()?;
        analysis_complete &= collections_complete;
        for collection in referenced_collections {
            live_raw_space.push(self.collection_span(collection)?);
        }
        let userblock_space = Vec::from_iter(
            (self.context.signature().get() > 0).then_some((0, self.context.signature().get())),
        );

        let superblock_len = u64::try_from(
            hdf5_pure_format::__private::serialize_superblock(self.context.superblock())?.len(),
        )
        .map_err(|_err| FormatError::Internal("superblock length does not fit u64".into()))?;
        live_metadata_space.push((self.context.signature().get(), superblock_len));

        if let Some(extension) = self.context.superblock().superblock_extension_address
            && !extension.is_undefined(self.widths().offsets)
        {
            let extension = self.context.addresses().absolute(extension)?;
            let header = ObjectHeader::parse_from_source_with_widths(
                &SourceMetadata(&self.image()),
                AccessMode::ReadWrite,
                extension,
                self.widths(),
                self.context.addresses().effective_base(),
            )?;
            live_metadata_space.extend(self.oh_chunk_spans(extension)?);
            analysis_complete &= header.messages.iter().all(|message| {
                !message.flags.is_shared()
                    && matches!(
                        message.msg_type,
                        MessageType::NIL
                            | MessageType::OBJECT_HEADER_CONTINUATION
                            | MessageType::FILE_SPACE_INFO
                            | MessageType::B_TREE_K_VALUES
                    )
            });
        }

        let mut free_space_manager_space = Vec::new();
        if let Some((_, info)) = Self::extension_fsinfo(&self.image(), &self.context)? {
            if info.persist {
                let widths = self.widths();
                let (_, blocks) = crate::free_space_manager::read_persisted_sections_source(
                    &self.image(),
                    widths,
                    self.context.addresses().effective_base(),
                    &info.manager_addrs,
                )?;
                free_space_manager_space.extend(blocks);
            }
        }

        let reusable_free_space = self.space_accounting()?.reusable_free_space;
        let mut claimed = userblock_space
            .iter()
            .chain(&live_raw_space)
            .chain(&live_metadata_space)
            .chain(&free_space_manager_space)
            .chain(&reusable_free_space)
            .copied()
            .collect::<Vec<_>>();
        claimed.sort_unstable_by_key(|&(addr, _)| addr);
        let mut previous_end = 0;
        for &(address, length) in &claimed {
            let end = address
                .checked_add(length)
                .ok_or(FormatError::OffsetOverflow {
                    offset: address,
                    length,
                })?;
            if address < previous_end || end > logical_size {
                return Err(Error::InvalidSpaceOwnershipExtent {
                    address,
                    length,
                    previous_end,
                    logical_size,
                });
            }
            previous_end = end;
        }

        let unclassified_space = complement_spans(logical_size, &claimed);
        let trailing_slack_space = (physical_size > logical_size)
            .then_some((logical_size, physical_size - logical_size))
            .into_iter()
            .collect();

        Ok(SpaceOwnership {
            physical_size,
            logical_size,
            userblock_space,
            live_raw_space,
            live_metadata_space,
            free_space_manager_space,
            reusable_free_space,
            unclassified_space,
            analysis_complete,
            trailing_slack_space,
        })
    }

    /// Returns the span `(address, length)` of the global heap collection at the absolute file
    /// offset `address`.
    ///
    /// The walk parses every object header of the collection, so a collection that is malformed
    /// past its header is an error.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`global_heap::walk_collection`].
    fn collection_span(&self, address: u64) -> Result<(u64, u64), FormatError> {
        let range = global_heap::walk_collection(
            self.image(),
            self.widths().lengths,
            PhysicalAddress::new(address),
            |_| {},
        )?;
        Ok((range.start().get(), range.length()))
    }

    /// Returns every chunk's full on-disk span as `(absolute address, length)`.
    ///
    /// The spans include the prefix, continuation signatures, and trailing checksums. The caller
    /// uses them to reclaim storage only after the complete traversal succeeds.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the header cannot be traversed. Propagates
    /// parsing and source-read errors from [`super::object_header::read_oh_chunks`].
    pub(super) fn oh_chunk_spans(&self, addr: u64) -> Result<Vec<(u64, u64)>, Error> {
        Ok(
            super::object_header::read_oh_chunks(&self.image(), addr, self.context.addresses())?
                .into_iter()
                .map(|chunk| chunk.span)
                .collect(),
        )
    }

    /// Returns the storage of the dense attribute set that the Attribute Info message in `region`
    /// names: its version 2 B-tree indexes, and its fractal heap where the heap's ownership is
    /// proven.
    ///
    /// The set is checked against the object header chunks in `header_spans`, as
    /// [`dense_storage_spans`](Self::dense_storage_spans) describes. An object without dense
    /// attributes has an empty set. A replacement header cannot overlap the returned spans: the
    /// commit gathers them in `to_free`, places every replacement, and only then adds `to_free` to
    /// the session's free space.
    ///
    /// Returns `None`, and leaves the whole set allocated, if `region` cannot be read, holds more
    /// than one Attribute Info message, or the indexes cannot be accounted for.
    pub(super) fn dense_attribute_storage_spans(
        &self,
        region: &OhRegion,
        header_spans: &[(u64, u64)],
    ) -> Option<Vec<(u64, u64, FreeClass)>> {
        let mut attr_info = None;
        let mut p = 0;
        while let Some((msg_type, body, body_end)) = region.next_message(p).ok()? {
            if msg_type == MessageType::ATTRIBUTE_INFO {
                if attr_info.is_some() {
                    return None;
                }
                attr_info = Some(
                    AttributeInfoMessage::parse_with_offset_width(
                        &region[body..body_end],
                        self.widths().offsets,
                    )
                    .ok()?,
                );
            }
            p = body_end;
        }
        let Some(attr_info) = attr_info else {
            return Some(Vec::new());
        };
        let Some(heap_address) = attr_info.fractal_heap_address else {
            return Some(Vec::new());
        };

        let index = crate::attribute::collect_dense_attribute_index_storage_extents(
            &BaseOffsetSource {
                inner: &self.image(),
                base: self.context.addresses().effective_base(),
            },
            &attr_info,
            self.widths(),
        )
        .ok()?;
        self.dense_storage_spans(header_spans, heap_address, &index)
    }

    /// Returns the storage of the dense link set that `link_info` names: its type 5 name index, its
    /// optional type 6 creation-order index, and its fractal heap where the heap's ownership is
    /// proven.
    ///
    /// The set is checked against the group's header chunks in `header_spans` and its dense
    /// attribute storage in `attribute_storage`, as
    /// [`dense_storage_spans`](Self::dense_storage_spans) describes. A group with compact link
    /// storage has an empty set.
    ///
    /// Returns `None`, and leaves the whole set allocated, if the indexes cannot be accounted for.
    pub(super) fn dense_group_storage_spans(
        &self,
        header_spans: &[(u64, u64)],
        attribute_storage: Option<&[(u64, u64, FreeClass)]>,
        link_info: &LinkInfoMessage,
    ) -> Option<Vec<(u64, u64, FreeClass)>> {
        let Some(heap_address) = link_info.fractal_heap_address else {
            return Some(Vec::new());
        };
        let index = crate::group_v2::collect_dense_group_index_storage_extents(
            &BaseOffsetSource {
                inner: &self.image(),
                base: self.context.addresses().effective_base(),
            },
            link_info,
            self.widths(),
        )
        .ok()?;
        let owned: Vec<(u64, u64)> = header_spans
            .iter()
            .copied()
            .chain(
                attribute_storage
                    .into_iter()
                    .flatten()
                    .map(|&(addr, len, _)| (addr, len)),
            )
            .collect();
        self.dense_storage_spans(&owned, heap_address, &index)
    }

    /// Returns the `index` blocks of a dense set as metadata, followed by the allocations of the
    /// fractal heap at `heap_address` where the heap's ownership is proven.
    ///
    /// The heap joins the set only if every allocation of it lies in the file and overlaps neither
    /// `owned` nor the index, and is left allocated otherwise.
    ///
    /// Returns `None` if an index block lies outside the file or overlaps another index block or a
    /// span in `owned`.
    fn dense_storage_spans(
        &self,
        owned: &[(u64, u64)],
        heap_address: StoredAddress,
        index: &[Extent],
    ) -> Option<Vec<(u64, u64, FreeClass)>> {
        let base = self.context.addresses();
        let eof = self.image.len();
        let index_spans = index
            .iter()
            .map(|extent| {
                Some((
                    base.absolute(StoredAddress::new(extent.start())).ok()?,
                    extent.len(),
                ))
            })
            .collect::<Option<Vec<_>>>()?;
        let mut proven = owned.to_vec();
        proven.extend(index_spans.iter().copied());
        if !super::spans_disjoint_in_bounds(&mut proven, eof) {
            return None;
        }

        let mut storage: Vec<(u64, u64, FreeClass)> = meta_spans(index_spans).collect();
        if let Some(heap_spans) = self.fractal_heap_spans(heap_address) {
            proven.extend(heap_spans.iter().map(|&(addr, len, _)| (addr, len)));
            if super::spans_disjoint_in_bounds(&mut proven, eof) {
                storage.extend(heap_spans);
            }
        }
        Some(storage)
    }

    /// Returns every allocation of the fractal heap at `heap_address` as an absolute span, with
    /// the free class [`fractal_heap_free_spans`] gives it.
    ///
    /// Returns `None` if
    /// [`collect_fractal_heap_storage_from_source`](
    ///     fractal_heap_storage::collect_fractal_heap_storage_from_source
    /// )
    /// cannot prove the heap's ownership, as for a filtered heap with a filter this crate does not
    /// decode or with huge objects, or for invalid manager metadata.
    fn fractal_heap_spans(
        &self,
        heap_address: StoredAddress,
    ) -> Option<Vec<(u64, u64, FreeClass)>> {
        let base = self.context.addresses();
        let storage = fractal_heap_storage::collect_fractal_heap_storage_from_source(
            &BaseOffsetSource {
                inner: &self.image(),
                base: base.effective_base(),
            },
            self.context.addresses(),
            heap_address,
            self.widths(),
        )
        .ok()?;
        let allocations: Vec<_> = storage
            .allocations()
            .iter()
            .map(|&(extent, kind)| (extent.start(), extent.len(), kind))
            .collect();
        fractal_heap_free_spans(self.space.page_size(), &allocations)
            .into_iter()
            .map(|(addr, len, class)| {
                Some((base.absolute(StoredAddress::new(addr)).ok()?, len, class))
            })
            .collect()
    }

    /// Count, for every object-header address reachable from the root, how many
    /// hard links in the *pre-commit* file point to it. The result drives the
    /// last-hard-link reclaim guard in [`collect_free_spans`](Self::collect_free_spans):
    /// an object is freed only when its count is 1.
    ///
    /// Walks the whole link graph from the root, following hard links through
    /// groups of any on-disk format (v0/v1 symbol-table, v2 compact, v2 dense)
    /// via [`group_v2::resolve_group_entries_from_source`], tallying each hard-link edge. Datasets and
    /// other leaves contribute no edges. Returns `None` if the graph cannot be walked in full:
    /// an unparseable header, a group whose links cannot be enumerated, or
    /// more than [`MAX_LINK_GRAPH_NODES`] objects. Cycles are handled by visiting
    /// each object once. Base-aware: stored child addresses are shifted by the
    /// userblock base, so the returned keys are absolute file offsets.
    pub(super) fn count_incoming_hard_links(&self) -> Option<HashMap<u64, u32>> {
        let base = self.context.addresses();
        let mut counts: HashMap<u64, u32> = HashMap::new();
        let mut visited: HashSet<u64> = HashSet::new();
        let mut stack: Vec<u64> = vec![
            self.context
                .addresses()
                .absolute(self.context.superblock().root_group_address)
                .ok()?,
        ];
        let mut budget = MAX_LINK_GRAPH_NODES;
        while let Some(addr) = stack.pop() {
            if !visited.insert(addr) {
                continue; // already expanded (also breaks hard-link cycles)
            }
            if budget == 0 {
                return None; // graph exceeds the walk limit, so leak conservatively
            }
            budget -= 1;
            let header = ObjectHeader::parse_from_source_with_widths(
                &SourceMetadata(&self.image()),
                AccessMode::ReadWrite,
                addr,
                self.widths(),
                base.effective_base(),
            )
            .ok()?;
            // Datasets and other leaves are not groups and own no links.
            let is_group = header.messages.iter().any(|m| {
                matches!(
                    m.msg_type,
                    MessageType::SYMBOL_TABLE | MessageType::LINK | MessageType::LINK_INFO
                )
            });
            if !is_group {
                continue;
            }
            // A group we cannot enumerate fully would undercount incoming links
            // and risk over-reclaim. Return the safe-leak fallback.
            let entries = group_v2::resolve_group_entries_from_source(
                &self.image(),
                AccessMode::ReadWrite,
                &header,
                self.widths(),
                self.context.addresses(),
                group_v1::group_leaf_node_k(self.context.superblock()),
            )
            .ok()?;
            for e in entries {
                let child = base.absolute(e.object_header_address).ok()?;
                *counts.entry(child).or_insert(0) += 1;
                stack.push(child);
            }
        }
        Some(counts)
    }

    /// Appends to `out` the span and free class of every block the object at `addr` owns, and for
    /// a group those of its whole subtree, for reclamation after a delete.
    ///
    /// The walk accounts for an object header, a contiguous dataset's data block, a chunked
    /// dataset's chunk index and chunks through
    /// [`chunked_storage_spans`](Self::chunked_storage_spans), a dense attribute set, and a group's
    /// links and subtree, compact or dense. An object it cannot account for in full, such as one
    /// with a header that is not version 2, a chunk index that cannot be walked in full, or a
    /// compact group with a soft or external link, contributes nothing and is not descended into,
    /// so `out` never holds a region still in use. A dense set's indexes join `out` only once the
    /// whole index set is accounted for, and its fractal heap only once the heap's ownership is
    /// proven. A group's link storage and the objects its links reach are separate: either can be
    /// reclaimed without the other. The walk excludes the global heap collections the objects
    /// refer to, since objects can share a collection, and the commit frees those it leaves
    /// unreferenced through [`unreferenced_collection_spans`](Self::unreferenced_collection_spans).
    /// The walk stops at [`MAX_COPY_DEPTH`] against a hard-link cycle.
    ///
    /// `incoming` holds the file-wide hard-link count of each object header address, from
    /// [`count_incoming_hard_links`](Self::count_incoming_hard_links). An object is reclaimed, and
    /// a group descended into, only when its count is 1, so deleting one of several hard links
    /// leaves the object and everything below it allocated.
    pub(super) fn collect_free_spans(
        &self,
        addr: u64,
        depth: u32,
        incoming: &HashMap<u64, u32>,
        out: &mut Vec<(u64, u64, FreeClass)>,
    ) {
        // `addr` is an absolute file offset (the caller resolves it from the live
        // file, and the group recursion below re-absolutizes each child). `incoming`
        // is keyed by absolute offset, and `oh_chunk_spans`/`chunked_storage_spans`
        // both take an absolute address and return absolute spans, so the whole
        // walk works in absolute file offsets. The one shift this method must apply
        // itself is on the stored addresses `read_object` returns for a contiguous
        // data block and a group's child links: each is converted to an absolute
        // offset (a no-op on a base-0 file) before it is bounds-checked, recorded,
        // or descended into.
        let base = self.context.addresses();
        let file_len = self.image().len();
        if depth >= MAX_COPY_DEPTH {
            return;
        }
        // Reclaim only when this delete removes the object's last hard link. A
        // count other than 1 (it has surviving links, or the graph walk could
        // not account for it) means the object and a group's whole subtree stay live and must
        // not be freed.
        if incoming.get(&addr) != Some(&1) {
            return;
        }
        // The header's own chunks. If they cannot be mapped, account for nothing.
        let spans = match self.oh_chunk_spans(addr) {
            Ok(s) => s,
            Err(_) => return,
        };
        let region = Self::gather_oh_messages(&self.image(), addr, self.context.addresses()).ok();
        let attribute_storage = region
            .as_ref()
            .and_then(|region| self.dense_attribute_storage_spans(region, &spans));
        if let Some((link_info, children)) = region
            .as_ref()
            .and_then(|region| self.dense_group_children(region))
        {
            let link_storage =
                self.dense_group_storage_spans(&spans, attribute_storage.as_deref(), &link_info);
            self.collect_group_spans(
                depth,
                incoming,
                &spans,
                attribute_storage.as_deref(),
                link_storage.as_deref(),
                children,
                out,
            );
            return;
        }
        match Self::read_object(
            &self.image(),
            AccessMode::ReadWrite,
            addr,
            self.context.addresses(),
        ) {
            Ok(ObjModel::DatasetVerbatim { .. }) => {
                self.append_object_metadata_spans(out, &spans, attribute_storage.as_deref());
            }
            Ok(ObjModel::DatasetContiguous {
                data_addr,
                data_size,
                ..
            }) => {
                self.append_object_metadata_spans(out, &spans, attribute_storage.as_deref());
                // A defined, in-bounds contiguous data block is owned outright.
                // An empty dataset stores the undefined address and owns none. Its
                // absolute file offset is what this bounds-checks and records.
                if !data_addr.is_undefined(self.widths().offsets) && data_size > 0 {
                    if let Ok(abs) = base.absolute(data_addr) {
                        if abs.checked_add(data_size).is_some_and(|e| e <= file_len) {
                            // A contiguous data block is raw data.
                            out.push((abs, data_size, FreeClass::Page(PageType::Raw)));
                        }
                    }
                }
            }
            Ok(ObjModel::Group { children, .. }) => {
                self.collect_group_spans(
                    depth,
                    incoming,
                    &spans,
                    attribute_storage.as_deref(),
                    None,
                    children.into_iter().map(|(_, _, child)| child),
                    out,
                );
            }
            // A chunked dataset: reclaim its chunk index and chunk data blocks
            // alongside its header. `chunked_storage_spans` returns `None` for
            // anything it cannot account for exhaustively (an index type with no
            // walker, an undefined index address, or spans that fail the
            // bounds/overlap check), leaving the whole dataset as dead bytes and not freeing a
            // region that might still be in use.
            Ok(ObjModel::DatasetChunked { .. }) => {
                if let Some(storage) = self.chunked_storage_spans(addr) {
                    self.append_object_metadata_spans(out, &spans, attribute_storage.as_deref());
                    // Already page-typed: chunk data is raw, and the index class follows its
                    // format allocation type or proven physical placement.
                    out.extend(storage);
                }
            }
            // A truly unsupported object (one `read_object` cannot model): leave
            // its bytes in place because its extent is unknown.
            //
            // An externally stored dataset lands here as of #336, where before it
            // was modelled as the contiguous dataset it structurally is and had
            // its header chunks reclaimed. Deleting one therefore leaves those
            // chunks behind: measured on the fixture in
            // `crates/crosscheck/tests/main/external_storage.rs`, a commit that deletes it
            // reports 147 B reusable where it reported 431 B. That is a leak and
            // not a hazard. `oh_chunk_spans` never covered the local heap the
            // External Data Files message names either, so neither the old
            // behaviour nor this one accounted for the whole object. This leak is
            // the price of stating the rejection once, in the one function both the
            // copy planner and this walk read objects through.
            Err(_) => {}
        }
    }

    /// Appends the storage a group owns to `out`, then collects the storage of each child in
    /// `children` one level deeper.
    fn collect_group_spans(
        &self,
        depth: u32,
        incoming: &HashMap<u64, u32>,
        header_spans: &[(u64, u64)],
        attribute_storage: Option<&[(u64, u64, FreeClass)]>,
        link_storage: Option<&[(u64, u64, FreeClass)]>,
        children: impl IntoIterator<Item = StoredAddress>,
        out: &mut Vec<(u64, u64, FreeClass)>,
    ) {
        self.append_object_metadata_spans(out, header_spans, attribute_storage);
        out.extend(link_storage.into_iter().flatten().copied());
        // The recursion works in absolute offsets, matching `incoming`'s
        // keys and `oh_chunk_spans`.
        let base = self.context.addresses();
        for child in children {
            if let Ok(c) = base.absolute(child) {
                self.collect_free_spans(c, depth + 1, incoming, out);
            }
        }
    }

    /// Appends classified header chunks and independently enumerated attribute storage to `out`.
    ///
    /// Skips the headers if any chunk address lies below the effective base. Attribute spans
    /// retain their supplied free classes.
    fn append_object_metadata_spans(
        &self,
        out: &mut Vec<(u64, u64, FreeClass)>,
        header_spans: &[(u64, u64)],
        attribute_storage: Option<&[(u64, u64, FreeClass)]>,
    ) {
        if let Ok(headers) = self.object_header_free_spans(header_spans) {
            out.extend(headers);
        }
        out.extend(attribute_storage.into_iter().flatten().copied());
    }

    /// Returns independently classified object-header chunks at their physical offsets.
    ///
    /// The caller passes complete chunks from [`Self::oh_chunk_spans`]. Classification uses
    /// stored addresses, while the returned spans retain their original physical offsets for
    /// the reclamation walk.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressBelowBase`] if a chunk starts below the effective base.
    pub(super) fn object_header_free_spans(
        &self,
        spans: &[(u64, u64)],
    ) -> Result<Vec<(u64, u64, FreeClass)>, FormatError> {
        spans
            .iter()
            .map(|&(address, length)| {
                let stored = self.context.addresses().relative(address)?;
                Ok((
                    address,
                    length,
                    self.space.object_header_free_class(stored.get(), length),
                ))
            })
            .collect()
    }

    /// Returns the Link Info message in `region` and the object header address of each hard link
    /// it indexes, for a group with dense link storage.
    ///
    /// Returns `None` if `region` lacks a Link Info message or holds more than one, if the group
    /// stores its links compactly, or if the links cannot be read.
    fn dense_group_children(
        &self,
        region: &OhRegion,
    ) -> Option<(LinkInfoMessage, Vec<StoredAddress>)> {
        let mut link_info = None;
        let mut p = 0;
        while let Some((msg_type, body, body_end)) = region.next_message(p).ok()? {
            if msg_type == MessageType::LINK_INFO {
                if link_info.is_some() {
                    return None;
                }
                link_info = Some(
                    LinkInfoMessage::parse_with_offset_width(
                        &region[body..body_end],
                        self.widths().offsets,
                    )
                    .ok()?,
                );
            }
            p = body_end;
        }
        let link_info = link_info?;
        let heap_address = link_info.fractal_heap_address?;
        let entries = crate::group_v2::resolve_dense_entries_from_source(
            &BaseOffsetSource {
                inner: &self.image(),
                base: self.context.addresses().effective_base(),
            },
            AccessMode::ReadWrite,
            self.context.addresses(),
            &link_info,
            heap_address,
            self.widths(),
        )
        .ok()?;
        Some((
            link_info,
            entries
                .into_iter()
                .map(|entry| entry.object_header_address)
                .collect(),
        ))
    }

    /// Returns the global heap collections that the objects at `deleted`, and the objects
    /// reachable from them, refer to.
    ///
    /// `deleted` holds the absolute addresses of the object headers whose links a commit deletes.
    /// The result is empty if [`collections_reachable_from`](Self::collections_reachable_from)
    /// returns an error or an incomplete analysis for them, and leaves out a collection that
    /// cannot be parsed, so the commit keeps that collection allocated.
    pub(super) fn deleted_collections(&self, deleted: &[u64]) -> DeletedCollections {
        let Ok(GlobalHeapLiveness {
            referenced_collections,
            analysis_complete: true,
        }) = self.collections_reachable_from(deleted.iter().copied())
        else {
            return DeletedCollections(Vec::new());
        };
        DeletedCollections(
            referenced_collections
                .into_iter()
                .filter_map(|address| self.collection_span(address).ok())
                .collect(),
        )
    }

    /// Returns the span and free class of each of the deleted `collections` that is unreferenced
    /// by the objects reachable from the object header at `root`.
    ///
    /// `root` is the absolute address of the root group header that the commit publishes. The
    /// result is empty if [`collections_reachable_from`](Self::collections_reachable_from) returns
    /// an error or an incomplete analysis for `root`, since an object the analysis cannot read may
    /// refer to any collection. Each span has the free class [`collection_free_class`] returns.
    pub(super) fn unreferenced_collection_spans(
        &self,
        root: u64,
        DeletedCollections(collections): DeletedCollections,
    ) -> Vec<(u64, u64, FreeClass)> {
        if collections.is_empty() {
            return Vec::new();
        }
        let Ok(GlobalHeapLiveness {
            referenced_collections,
            analysis_complete: true,
        }) = self.collections_reachable_from([root])
        else {
            return Vec::new();
        };
        collections
            .into_iter()
            .filter(|(address, _)| !referenced_collections.contains(address))
            .filter_map(|(address, len)| {
                let stored = self.context.addresses().relative(address).ok()?;
                Some((
                    address,
                    len,
                    collection_free_class(self.space.page_size(), stored.get(), len),
                ))
            })
            .collect()
    }

    /// Returns every on-disk block a *chunked* dataset at `addr` owns: its
    /// chunk index structure (B-tree v1 nodes, or fixed- / extensible-array
    /// header, index, super, and data blocks) plus every allocated chunk data
    /// block.
    ///
    /// The object-header chunks are freed by the caller
    /// ([`collect_free_spans`](Self::collect_free_spans)). This returns only the
    /// storage the data-layout message points at.
    ///
    /// Returns `None` and contributes nothing, leaving the object as dead bytes whenever the
    /// dataset cannot be enumerated *exhaustively* and safely: a
    /// header that does not parse or is not a chunked dataset, an undefined index address (an
    /// empty, never-written dataset), or any resulting span that
    /// falls outside the file image or overlaps another. This upholds the
    /// editor's invariant that reclaimed space is never a region still in use:
    /// under-reclaiming only wastes space, while over-reclaiming would corrupt.
    ///
    /// Chunk data addresses and sizes come from the same index walkers the
    /// reader uses, so they match the bytes the writer laid down exactly. The
    /// per-layout enumeration lives in
    /// [`chunked_read::collect_chunked_storage_spans`].
    /// This method only locates the layout and dataspace messages and validates
    /// the result. The global heap collections that the chunks refer to are
    /// excluded, as in [`collect_free_spans`](Self::collect_free_spans).
    pub(super) fn chunked_storage_spans(&self, addr: u64) -> Option<Vec<(u64, u64, FreeClass)>> {
        let split = self.chunked_storage_parts(addr)?;
        let index_class = chunk_index_free_class(
            split.index_layout,
            self.index_is_provably_raw(&split.data, &split.index),
        );
        let mut spans: Vec<(u64, u64, FreeClass)> = split
            .data
            .iter()
            .map(|&(a, l)| (a, l, FreeClass::Page(PageType::Raw)))
            .collect();
        spans.extend(split.index.into_iter().map(|(a, l)| (a, l, index_class)));
        Some(spans)
    }

    /// Returns the raw chunk data and metadata index blocks of a chunked dataset separately.
    fn chunked_storage_parts(&self, addr: u64) -> Option<ChunkedStorageParts> {
        // Locate the data-layout and dataspace messages in the object header.
        let region =
            Self::gather_oh_messages(&self.image(), addr, self.context.addresses()).ok()?;
        let mut layout_msg: Option<(usize, usize)> = None;
        let mut dataspace_msg: Option<(usize, usize)> = None;
        let mut p = 0;
        loop {
            match region.next_message(p) {
                Ok(Some((msg_type, body, body_end))) => {
                    match msg_type {
                        MessageType::DATA_LAYOUT => layout_msg = Some((body, body_end)),
                        MessageType::DATASPACE => dataspace_msg = Some((body, body_end)),
                        _ => {}
                    }
                    p = body_end;
                }
                Ok(None) => break,
                Err(_) => return None,
            }
        }
        let (lb, le) = layout_msg?;
        let (db, de) = dataspace_msg?;

        let layout = DataLayout::parse_with_widths(&region[lb..le], self.widths()).ok()?;
        if !matches!(layout, DataLayout::Chunked { .. }) {
            return None;
        }
        let dataspace =
            Dataspace::parse_with_length_width(&region[db..de], self.widths().lengths).ok()?;

        // Delegate the per-index-type enumeration to the chunked reader (the
        // single owner of chunk-storage layout knowledge), then validate: every
        // span must lie inside the current file image and be pairwise disjoint,
        // or the free list would later hand out live bytes (and a debug build
        // would panic on the double-free). On any error or violation, leave the
        // whole dataset unreclaimed because freeing a region still in use is unsafe.
        //
        // The layout's stored addresses are relative to the userblock base, so the
        // enumeration runs on a base-relative view of the file and each returned
        // span address is shifted back to an absolute file offset by adding `base`
        // (a no-op on a base-0 file). The free list and the bounds check below both
        // work in absolute file offsets.
        let base = self.context.addresses();
        let split = crate::chunked_read::collect_chunked_storage_spans(
            &BaseOffsetSource {
                inner: &self.image(),
                base: base.effective_base(),
            },
            &layout,
            &dataspace,
            self.widths(),
        )
        .ok()?;
        // Chunk data is raw under every writer, so its spans are raw outright. A version 2
        // B-tree is metadata by the format's allocation taxonomy. Other index structures are raw
        // only where this crate provably placed them beside their chunk data. An index with an
        // unproven PAGE type is recorded as dead.
        let mut data: Vec<(u64, u64)> = Vec::with_capacity(split.data.len());
        for (addr, len) in split.data {
            data.push((base.absolute(StoredAddress::new(addr)).ok()?, len));
        }
        let mut index: Vec<(u64, u64)> = Vec::with_capacity(split.index.len());
        for (addr, len) in split.index {
            index.push((base.absolute(StoredAddress::new(addr)).ok()?, len));
        }
        // Validate both halves together. They must be disjoint from each other and internally
        // before either is trusted, including by the proof.
        let mut plain: Vec<(u64, u64)> = data.iter().chain(index.iter()).copied().collect();
        if !super::spans_disjoint_in_bounds(&mut plain, self.image.len()) {
            return None;
        }
        let index_layout = match layout {
            DataLayout::Chunked { index, .. } => index,
            _ => return None,
        };
        Some(ChunkedStorageParts {
            data,
            index,
            index_layout,
        })
    }

    /// Whether a chunked dataset's `index` provably occupies raw pages, given the
    /// `data` spans it accompanies. Both are absolute file offsets.
    ///
    /// A chunk index is *metadata* by the format's taxonomy, and the reference C
    /// library allocates one as such, out of metadata pages and nowhere near the
    /// chunk data. This crate instead lays a dataset's index down in the same run
    /// as its chunk data, inside raw pages, so that a reader following the layout
    /// message walks one contiguous blob. Both are valid. On a paged
    /// file is that freeing an index records it under the page type it actually
    /// sits in, since that decides what a later allocation may overwrite there.
    ///
    /// Nothing in the file says which writer produced it, so the index is placed
    /// from the layout itself, by two tests it must pass together: some chunk-data
    /// span must **abut** it, ending exactly where the index begins or beginning
    /// exactly where it ends. No byte of it may lie in **page 0**.
    ///
    /// The abutment is this crate's single blob, and is not a layout the reference
    /// library sets out to produce. A chunk index is metadata to that library,
    /// allocated out of metadata pages. Those pages are allocated from the bottom
    /// of the file, while large raw data goes above, so its index lands far below
    /// the chunk data it indexes, with a gap between them. Measured across seven
    /// files it wrote (Extensible and Fixed Array indexes, page sizes
    /// 512/4096/8192, 8 to 16 chunks): the index began in page 0 every time, with
    /// the chunk data starting at page 9 or higher, so neither join address was
    /// within reach of it.
    ///
    /// The page-0 screen closes the one way that measured layout can nonetheless
    /// satisfy the abutment. An index block that ends exactly on a page boundary
    /// with a raw page after it, for example a Fixed Array data block that exactly fills its
    /// page, is abutted by the first chunk in that raw page, and the whole
    /// index run would then be filed as raw, its header included, *even though
    /// that header sits in page 0 beside the superblock*. A later raw allocation
    /// would land inside a metadata page, which is the page-mixing defect of
    /// issue #261. Page 0 of a paged file always begins with the superblock, so it
    /// is a metadata page by construction and an index with a byte in it cannot be
    /// raw whatever abuts it. The screen costs this crate nothing: it allocates
    /// chunk data and the indexes beside it out of raw pages. Page 0 is never wholly free because
    /// the superblock lives there, so it is never one of them.
    ///
    /// The tighter rule considered instead was to require the join address to fall
    /// *strictly inside* a page, which proves rawness outright: the byte before
    /// the join and the byte after it are then in one page, and a page holding
    /// chunk data is raw. It is sound, and a paged file the reference library
    /// wrote cannot satisfy it at all, since such a file never puts raw and
    /// metadata bytes in one page and so can only ever join them on a boundary.
    /// It was rejected because it also rejects layouts *this* crate produces:
    /// whenever a blob's chunk bytes total a page multiple from a page-aligned
    /// start, its index begins on a boundary too. Measured over 100
    /// `append_staged` commits at a 4096-byte page with 2048-byte chunks, that
    /// stranded about 500 bytes per commit, compared with the 10 KB the abutment rule left. The
    /// two writers produce the same geometry there and
    /// nothing in the file distinguishes them.
    ///
    /// Both sides have to be admitted, not just the index that follows its data.
    /// A staged append keeps the existing chunks where they are and places the new
    /// ones past the index it is superseding, so from the second append onward the
    /// dataset has chunk data on both sides of that index and the highest data
    /// address is no longer the one beside it. A predicate that checks only whether the index
    /// begins where the *last* chunk ends rejected every such index, and a
    /// repeatedly appended dataset dropped one per commit (issue #388).
    ///
    /// An index the test does not place is [dead](FreeClass::Dead) to the caller. Its bytes are
    /// no longer in use, but advertising space
    /// inside a possible metadata page for raw reuse would mix the page
    /// (issue #261), so they are reusable only once the whole page around them is
    /// empty.
    ///
    /// A non-paged file has no page types to keep apart, so everything is
    /// reclaimable there. The tag is ignored by its commit tail entirely.
    pub(super) fn index_is_provably_raw(&self, data: &[(u64, u64)], index: &[(u64, u64)]) -> bool {
        match self.space.page_size() {
            None => true,
            Some(page_size) => {
                index_abuts_chunk_data(data, index)
                    && index.iter().all(|&(at, len)| {
                        self.context.addresses().relative(at).is_ok_and(|stored| {
                            !touches_page_zero(page_size, [(stored.get(), len)])
                        })
                    })
            }
        }
    }

    /// Returns every on-disk byte span of a chunked dataset's *index structure
    /// only* (not its chunk data), for reclaiming the old index after a
    /// relocating append ([`super::MovingWrite::AppendedChunks`]) that keeps the chunk
    /// data in place.
    ///
    /// This is the index-only counterpart of
    /// [`chunked_storage_spans`](Self::chunked_storage_spans), delegating to
    /// [`chunked_read::chunk_index_spans_from_source`], which enumerates the index structure's
    /// own blocks (B-tree v1/v2 nodes, or fixed- / extensible-array header, index,
    /// super, and data blocks) and never a chunk-data address, so the shared
    /// kept chunk data is never freed. Base-aware and validated
    /// disjoint/in-bounds. Returns `None` (leave unreclaimed) on any error or
    /// violation.
    pub(super) fn chunked_index_spans(&self, addr: u64) -> Option<Vec<(u64, u64)>> {
        let region =
            Self::gather_oh_messages(&self.image(), addr, self.context.addresses()).ok()?;
        let mut layout_msg: Option<(usize, usize)> = None;
        let mut p = 0;
        loop {
            match region.next_message(p) {
                Ok(Some((msg_type, body, body_end))) => {
                    if msg_type == MessageType::DATA_LAYOUT {
                        layout_msg = Some((body, body_end));
                    }
                    p = body_end;
                }
                Ok(None) => break,
                Err(_) => return None,
            }
        }
        let (lb, le) = layout_msg?;
        let layout = DataLayout::parse_with_widths(&region[lb..le], self.widths()).ok()?;
        if !matches!(layout, DataLayout::Chunked { .. }) {
            return None;
        }
        let base = self.context.addresses();
        let mut spans = chunked_read::chunk_index_spans_from_source(
            &BaseOffsetSource {
                inner: &self.image(),
                base: base.effective_base(),
            },
            &layout,
            self.widths(),
        )
        .ok()?;
        for (a, _) in &mut spans {
            *a = base.absolute(StoredAddress::new(*a)).ok()?;
        }
        if !super::spans_disjoint_in_bounds(&mut spans, self.image.len()) {
            return None;
        }
        Some(spans)
    }
    /// Appends reachable raw and metadata extents, returning whether traversal is complete.
    ///
    /// Each object is visited once. Unsupported storage and traversal limits are reported
    /// through the return value, and parser failures are returned as errors.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if supported metadata cannot be parsed. Returns
    /// [`Error::MissingMessage`] if a chunked dataset lacks its dataspace. Returns
    /// [`Error::InvalidFractalHeapStorage`] if a heap has inconsistent allocation geometry.
    fn collect_owned_spans(
        &self,
        root: u64,
        raw: &mut Vec<(u64, u64)>,
        metadata: &mut Vec<(u64, u64)>,
    ) -> Result<bool, Error> {
        self.walk_reachable_objects([root], |addr, header, child| {
            if header.version != OWNED_HEADER_VERSION {
                return Ok(false);
            }
            metadata.extend(self.oh_chunk_spans(addr)?);
            self.collect_object_owned_spans(header, raw, metadata, child)
        })
    }

    /// Visits each object reachable from the object headers in `roots` once, and returns whether
    /// the walk is complete.
    ///
    /// `visit` receives the absolute address of an object header, the parsed header, and a
    /// callback that adds a child object to the walk by the absolute address of its header. The
    /// walk is incomplete if `visit` returns `false` for an object, or if the walk skips an object
    /// [`MAX_COPY_DEPTH`] links from a root or after visiting [`MAX_LINK_GRAPH_NODES`] objects.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if an object header is malformed, and any error `visit` returns.
    pub(super) fn walk_reachable_objects(
        &self,
        roots: impl IntoIterator<Item = u64>,
        mut visit: impl FnMut(u64, &ObjectHeader, &mut dyn FnMut(u64)) -> Result<bool, Error>,
    ) -> Result<bool, Error> {
        let mut complete = true;
        let mut visited = HashSet::new();
        let mut pending: Vec<(u64, u32)> = roots.into_iter().map(|root| (root, 0)).collect();
        while let Some((addr, depth)) = pending.pop() {
            if visited.contains(&addr) {
                continue;
            }
            if depth >= MAX_COPY_DEPTH || visited.len() >= MAX_LINK_GRAPH_NODES as usize {
                complete = false;
                continue;
            }
            visited.insert(addr);
            let header = ObjectHeader::parse_from_source_with_widths(
                &SourceMetadata(&self.image()),
                AccessMode::ReadWrite,
                addr,
                self.widths(),
                self.context.addresses().effective_base(),
            )?;
            complete &= visit(addr, &header, &mut |child| pending.push((child, depth + 1)))?;
        }
        Ok(complete)
    }

    /// Appends storage identified in the header and passes hard-link targets to `child`.
    ///
    /// Returns `false` for shared or unrecognized messages, or storage the layout, link, or
    /// attribute collectors cannot classify.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if supported metadata cannot be parsed. Returns
    /// [`Error::MissingMessage`] if a chunked dataset lacks its dataspace. Returns
    /// [`Error::InvalidFractalHeapStorage`] if a heap has inconsistent allocation geometry.
    fn collect_object_owned_spans(
        &self,
        header: &ObjectHeader,
        raw: &mut Vec<(u64, u64)>,
        metadata: &mut Vec<(u64, u64)>,
        mut child: impl FnMut(u64),
    ) -> Result<bool, Error> {
        let mut complete = true;
        let messages = &header.messages;
        for HeaderMessage {
            msg_type,
            flags,
            data,
            ..
        } in messages
        {
            if flags.is_shared() {
                complete = false;
                continue;
            }
            match *msg_type {
                MessageType::DATA_LAYOUT => {
                    complete &= self.collect_layout_owned_spans(messages, data, raw, metadata)?
                }
                MessageType::LINK => {
                    let link = LinkMessage::parse_with_offset_width(data, self.widths().offsets)?;
                    if let LinkTarget::Hard {
                        object_header_address,
                    } = link.link_target
                    {
                        child(self.context.addresses().absolute(object_header_address)?);
                    }
                }
                MessageType::LINK_INFO => {
                    complete &= self.collect_group_owned_spans(data, metadata, &mut child)?
                }
                MessageType::ATTRIBUTE_INFO => {
                    complete &= self.collect_attribute_owned_spans(data, metadata)?
                }
                MessageType::NIL
                | MessageType::DATASPACE
                | MessageType::DATATYPE
                | MessageType::FILL_VALUE_OLD
                | MessageType::FILL_VALUE
                | MessageType::GROUP_INFO
                | MessageType::FILTER_PIPELINE
                | MessageType::ATTRIBUTE
                | MessageType::OBJECT_HEADER_CONTINUATION
                | MessageType::OBJECT_MODIFICATION_TIME
                | MessageType::OBJECT_REFERENCE_COUNT => {}
                _ => complete = false,
            }
        }
        Ok(complete)
    }

    /// Appends dataset payload and chunk-index extents identified by the layout.
    ///
    /// Returns `false` for virtual storage or a chunked dataset with a shared dataspace.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the layout, dataspace, or chunk storage cannot be parsed.
    /// Returns [`Error::MissingMessage`] if a chunked dataset lacks its dataspace.
    fn collect_layout_owned_spans(
        &self,
        messages: &[HeaderMessage],
        data: &[u8],
        raw: &mut Vec<(u64, u64)>,
        metadata: &mut Vec<(u64, u64)>,
    ) -> Result<bool, Error> {
        let mut complete = true;
        let layout = DataLayout::parse_with_widths(data, self.widths())?;
        match &layout {
            DataLayout::Compact { .. } => {}
            DataLayout::Contiguous { address, size } => {
                if let Some(address) = address
                    && *size > 0
                {
                    raw.push((self.context.addresses().absolute(*address)?, *size));
                }
            }
            DataLayout::Chunked { .. } => {
                let space = messages
                    .iter()
                    .find(|m| m.msg_type == MessageType::DATASPACE)
                    .ok_or(Error::MissingMessage(MessageType::DATASPACE))?;
                if space.flags.is_shared() {
                    return Ok(false);
                }
                let dataspace =
                    Dataspace::parse_with_length_width(&space.data, self.widths().lengths)?;
                let storage = crate::chunked_read::collect_chunked_storage_spans(
                    &BaseOffsetSource {
                        inner: &self.image(),
                        base: self.context.addresses().effective_base(),
                    },
                    &layout,
                    &dataspace,
                    self.widths(),
                )?;
                raw.extend(self.absolute_owned_spans(storage.data)?);
                metadata.extend(self.absolute_owned_spans(storage.index)?);
            }
            DataLayout::Virtual => complete = false,
        }

        Ok(complete)
    }

    /// Appends dense link storage and passes absolute hard-link targets to `child`.
    ///
    /// Returns `false` when the group's heap ownership is unsupported.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the link information, index, heap, or entries cannot be parsed.
    /// Returns [`Error::InvalidFractalHeapStorage`] if the heap has inconsistent allocation geometry.
    fn collect_group_owned_spans(
        &self,
        data: &[u8],
        metadata: &mut Vec<(u64, u64)>,
        mut child: impl FnMut(u64),
    ) -> Result<bool, Error> {
        let info = LinkInfoMessage::parse_with_offset_width(data, self.widths().offsets)?;
        if let Some(heap) = info.fractal_heap_address {
            let source = BaseOffsetSource {
                inner: &self.image(),
                base: self.context.addresses().effective_base(),
            };
            let index = crate::group_v2::collect_dense_group_index_storage_extents(
                &source,
                &info,
                self.widths(),
            )?;
            metadata.extend(
                self.absolute_owned_spans(
                    index.iter().map(|extent| (extent.start(), extent.len())),
                )?,
            );
            if !self.collect_owned_heap(heap, metadata)? {
                return Ok(false);
            }
            let entries = crate::group_v2::resolve_dense_entries_from_source(
                &source,
                AccessMode::ReadWrite,
                self.context.addresses(),
                &info,
                heap,
                self.widths(),
            )?;
            for entry in entries {
                child(
                    self.context
                        .addresses()
                        .absolute(entry.object_header_address)?,
                );
            }
        }

        Ok(true)
    }

    /// Appends dense-attribute storage and returns whether it can be classified completely.
    ///
    /// Returns `false` if the ownership of the fractal heap is unsupported or a record in the
    /// storage is shared.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if supported attribute metadata cannot be parsed. Returns
    /// [`Error::InvalidFractalHeapStorage`] if the heap has inconsistent allocation geometry.
    fn collect_attribute_owned_spans(
        &self,
        data: &[u8],
        metadata: &mut Vec<(u64, u64)>,
    ) -> Result<bool, Error> {
        let info = AttributeInfoMessage::parse_with_offset_width(data, self.widths().offsets)?;
        let Some(heap) = info.fractal_heap_address else {
            return Ok(true);
        };
        let source = BaseOffsetSource {
            inner: &self.image(),
            base: self.context.addresses().effective_base(),
        };
        let index = crate::attribute::collect_dense_attribute_index_storage_extents(
            &source,
            &info,
            self.widths(),
        )?;
        metadata.extend(
            self.absolute_owned_spans(index.iter().map(|extent| (extent.start(), extent.len())))?,
        );
        if !self.collect_owned_heap(heap, metadata)? {
            return Ok(false);
        }
        let mut shared_records = false;
        crate::attribute::visit_dense_attribute_messages_from_source(
            &source,
            AccessMode::ReadWrite,
            self.context.addresses(),
            &info,
            heap,
            self.widths(),
            |flags| {
                shared_records |= flags.is_shared();
                !flags.is_shared()
            },
            |_data, _creation_index| Ok(()),
        )?;
        Ok(!shared_records)
    }

    fn absolute_owned_spans(
        &self,
        spans: impl IntoIterator<Item = (u64, u64)>,
    ) -> Result<Vec<(u64, u64)>, Error> {
        spans
            .into_iter()
            .map(|(addr, len)| {
                Ok((
                    self.context
                        .addresses()
                        .absolute(StoredAddress::new(addr))?,
                    len,
                ))
            })
            .collect()
    }

    /// Appends every heap allocation, returning whether ownership is supported.
    ///
    /// Returns `false` for a filtered heap with a filter this crate does not decode or with huge
    /// objects.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the heap cannot be parsed. Returns
    /// [`Error::InvalidFractalHeapStorage`] if the heap has inconsistent allocation geometry.
    fn collect_owned_heap(
        &self,
        heap: StoredAddress,
        metadata: &mut Vec<(u64, u64)>,
    ) -> Result<bool, Error> {
        let storage = match fractal_heap_storage::collect_fractal_heap_storage_from_source(
            &BaseOffsetSource {
                inner: &self.image(),
                base: self.context.addresses().effective_base(),
            },
            self.context.addresses(),
            heap,
            self.widths(),
        ) {
            Ok(storage) => storage,
            Err(
                FractalHeapStorageError::FilteredHugeObjects
                | FractalHeapStorageError::UnsupportedFilter(_),
            ) => return Ok(false),
            Err(FractalHeapStorageError::Format(error)) => return Err(error.into()),
            Err(FractalHeapStorageError::Layout(_) | FractalHeapStorageError::InvalidStorage) => {
                return Err(Error::InvalidFractalHeapStorage {
                    address: heap.get(),
                });
            }
        };
        for &(extent, _) in storage.allocations() {
            metadata.push((
                self.context
                    .addresses()
                    .absolute(StoredAddress::new(extent.start()))?,
                extent.len(),
            ));
        }
        Ok(true)
    }
}

/// Returns the PAGE free-space class of a chunk index after its physical placement is known.
///
/// Version 2 B-tree headers and nodes are metadata allocations. The other index structures this
/// editor emits share raw pages with their chunks where `provably_raw` establishes that placement.
/// An index with no established page type remains dead until its whole page becomes free.
pub(super) fn chunk_index_free_class(index: ChunkIndexLayout, provably_raw: bool) -> FreeClass {
    match index {
        ChunkIndexLayout::BTreeV2 { .. } => FreeClass::Page(PageType::Meta),
        _ if provably_raw => FreeClass::Page(PageType::Raw),
        _ => FreeClass::Dead,
    }
}

/// Whether some `data` span abuts the run of `index` blocks, ending exactly where
/// the index begins or beginning exactly where it ends. The geometric half of
/// [`WriteEngine::index_is_provably_raw`] contains the reasoning. This helper keeps the rule
/// expressible against spans alone.
///
/// An empty index is vacuously placed: there is nothing to file.
pub(super) fn index_abuts_chunk_data(data: &[(u64, u64)], index: &[(u64, u64)]) -> bool {
    if index.is_empty() {
        return true;
    }
    // The index as a whole: this crate writes its blocks as one run beside the
    // chunk data, so a data span abutting either end places all of them.
    let (Some(index_start), Some(index_end)) = (
        index.iter().map(|&(a, _)| a).min(),
        index.iter().filter_map(|&(a, l)| a.checked_add(l)).max(),
    ) else {
        return false;
    };
    // A zero-length span touches an address without occupying a byte beside it,
    // so it places nothing.
    data.iter().any(|&(addr, len)| {
        len > 0 && (addr.checked_add(len) == Some(index_start) || addr == index_end)
    })
}

/// Returns the PAGE free-space class of a global heap collection of `len` bytes at the stored
/// address `stored` that a delete frees.
///
/// A collection is raw data. On a paged file, a collection with a byte in page 0 or of a page or
/// more is [dead](FreeClass::Dead).
///
/// `H5HG_remove` (`H5HG.c`, libhdf5 2.2.0) frees a collection whole once its last object is
/// removed, as `H5FD_MEM_GHEAP` space, which the `H5FD_FLMAP_DICHOTOMY` free-list map
/// (`H5FDdevelop.h`) files with raw data. On a paged file, `H5MF_xfree` (`H5MF.c`, libhdf5 2.2.0)
/// files freed space of a page or more in the generic large manager by its size alone, whatever
/// its type.
fn collection_free_class(page_size: Option<FileSpacePageSize>, stored: u64, len: u64) -> FreeClass {
    match page_size {
        Some(page_size) if touches_page_zero(page_size, [(stored, len)]) => FreeClass::Dead,
        Some(page_size) if len >= page_size.get() => FreeClass::Dead,
        _ => FreeClass::Page(PageType::Raw),
    }
}

/// Returns whether any byte of `spans`, in stored addresses, lies in page 0 of a paged file.
///
/// Page 0 begins at the base address with the superblock, so it is a metadata page.
///
/// [`WriteEngine::index_is_provably_raw`], whose doc gives the reasoning, screens a chunk index
/// with it. [`huge_object_free_class`] screens a run of huge objects with it, and
/// [`collection_free_class`] a global heap collection.
pub(super) fn touches_page_zero(
    page_size: FileSpacePageSize,
    spans: impl IntoIterator<Item = (u64, u64)>,
) -> bool {
    spans
        .into_iter()
        .any(|(addr, len)| len > 0 && addr < page_size.get())
}

/// Tags each fractal heap allocation with the free class of the page it occupies.
///
/// `allocations` holds stored addresses sorted by address, as [`FractalHeapStorage::allocations`]
/// returns them. Without `page_size`, each allocation takes
/// [`FractalHeapAllocation::allocation_class`]. On a paged file an allocation of a page or more that
/// begins on a page boundary is dead, since the C library allocates it as whole pages of neither
/// type. Of the rest, the header, the managed blocks, and the huge-object index are metadata, and
/// a huge object takes the class [`huge_object_free_class`] gives the run of adjoining allocations
/// around it.
///
/// [`FractalHeapStorage::allocations`]: crate::fractal_heap_storage::FractalHeapStorage::allocations
fn fractal_heap_free_spans(
    page_size: Option<FileSpacePageSize>,
    allocations: &[(u64, u64, FractalHeapAllocation)],
) -> Vec<(u64, u64, FreeClass)> {
    let Some(page_size) = page_size else {
        return allocations
            .iter()
            .map(|&(addr, len, kind)| (addr, len, FreeClass::Page(kind.allocation_class().into())))
            .collect();
    };
    allocations
        .chunk_by(|&(addr, len, _), &(next, _, _)| addr.checked_add(len) == Some(next))
        .flat_map(|run| {
            let huge_object_class = huge_object_free_class(page_size, run);
            run.iter().map(move |&(addr, len, kind)| {
                let class = match kind {
                    // `H5MF__alloc_to_fs_type` (`H5MF.c`, libhdf5 2.2.0) gives an allocation of a
                    // page or more to the generic large manager whatever its type, and
                    // `H5MF__alloc_pagefs` starts it on a page boundary and files the rest of its
                    // last page in that manager, so no small allocation belongs in that page.
                    _ if len >= page_size.get() && addr % page_size.get() == 0 => FreeClass::Dead,
                    FractalHeapAllocation::Header
                    | FractalHeapAllocation::DirectBlock
                    | FractalHeapAllocation::IndirectBlock
                    | FractalHeapAllocation::HugeObjectIndex
                    | FractalHeapAllocation::ManagedFsmHeader
                    | FractalHeapAllocation::ManagedFsmSectionList => {
                        FreeClass::Page(kind.allocation_class().into())
                    }
                    FractalHeapAllocation::HugeObject => huge_object_class,
                };
                (addr, len, class)
            })
        })
        .collect()
}

/// Returns the free class of the huge objects in `run`, fractal heap allocations that each begin
/// where the one before ends.
///
/// A run of huge objects alone is raw data, or dead if a byte of it lies in page 0, which holds
/// the superblock. A huge object that adjoins heap metadata inside a page shares that
/// metadata page, and every huge object of its run is metadata. A run whose huge objects adjoin
/// heap metadata only on page boundaries is dead, as the page type of those huge objects is
/// unproven.
fn huge_object_free_class(
    page_size: FileSpacePageSize,
    run: &[(u64, u64, FractalHeapAllocation)],
) -> FreeClass {
    let is_huge_object = |&(_, _, kind): &(u64, u64, FractalHeapAllocation)| {
        kind == FractalHeapAllocation::HugeObject
    };
    if run.iter().all(is_huge_object) {
        return if touches_page_zero(page_size, run.iter().map(|&(addr, len, _)| (addr, len))) {
            FreeClass::Dead
        } else {
            FreeClass::Page(FractalHeapAllocation::HugeObject.allocation_class().into())
        };
    }
    // `H5MF__alloc_pagefs` (`H5MF.c`, libhdf5 2.2.0) keeps small raw-data and metadata
    // allocations in separate pages and starts each large allocation on a page boundary, so a
    // huge object joined to heap metadata inside a page shares an allocation with that metadata,
    // as in the blob `place_dense_attrs` writes.
    let joined_inside_a_page = run.windows(2).any(|pair| {
        is_huge_object(&pair[0]) != is_huge_object(&pair[1]) && pair[1].0 % page_size.get() != 0
    });
    if joined_inside_a_page {
        FreeClass::Page(PageType::Meta)
    } else {
        FreeClass::Dead
    }
}

/// Returns the gaps in `0..limit` left by sorted, disjoint `claimed` spans.
fn complement_spans(limit: u64, claimed: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let mut cursor = 0u64;
    for &(addr, len) in claimed {
        if cursor < addr {
            out.push((cursor, addr - cursor));
        }
        cursor = addr + len;
    }
    if cursor < limit {
        out.push((cursor, limit - cursor));
    }
    out
}

/// Tags each span as metadata for reclamation.
pub(super) fn meta_spans(spans: Vec<(u64, u64)>) -> impl Iterator<Item = (u64, u64, FreeClass)> {
    spans
        .into_iter()
        .map(|(a, l)| (a, l, FreeClass::Page(PageType::Meta)))
}

const OWNED_HEADER_VERSION: u8 = 2;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::huge_objects_apart_from_metadata_are_raw(
        &[(0x1000, 0x100, HEADER), (0x1200, 0x200, DIRECT), (0x3000, 0x80, HUGE), (0x3080, 0x40, HUGE)],
        &[(0x1000, 0x100, META), (0x1200, 0x200, META), (0x3000, 0x80, RAW), (0x3080, 0x40, RAW)],
    )]
    #[case::a_huge_object_in_page_zero_is_dead(
        &[(0x0f80, 0x100, HUGE), (0x1000, 0x100, HEADER)],
        &[(0x0f80, 0x100, FreeClass::Dead), (0x1000, 0x100, META)],
    )]
    #[case::huge_objects_joined_to_metadata_inside_a_page_are_metadata(
        &[(0x1000, 0x100, HEADER), (0x1100, 0x200, INDEX), (0x1300, 0x80, HUGE), (0x1380, 0x2000, HUGE)],
        &[(0x1000, 0x100, META), (0x1100, 0x200, META), (0x1300, 0x80, META), (0x1380, 0x2000, META)],
    )]
    #[case::a_huge_object_joined_to_metadata_on_a_page_boundary_is_dead(
        &[(0x1000, 0x100, HEADER), (0x1e00, 0x200, INDEX), (0x2000, 0x80, HUGE)],
        &[(0x1000, 0x100, META), (0x1e00, 0x200, META), (0x2000, 0x80, FreeClass::Dead)],
    )]
    #[case::page_aligned_allocations_of_a_page_or_more_are_dead(
        &[(0x1000, 0x100, HEADER), (0x2000, 0x1000, DIRECT), (0x4000, 0x1800, HUGE), (0x6000, 0x1000, INDEX)],
        &[(0x1000, 0x100, META), (0x2000, 0x1000, FreeClass::Dead), (0x4000, 0x1800, FreeClass::Dead), (0x6000, 0x1000, FreeClass::Dead)],
    )]
    #[case::allocations_of_a_page_or_more_joined_to_metadata_off_a_page_boundary_are_metadata(
        &[(0x1000, 0x100, HEADER), (0x1100, 0x1000, DIRECT), (0x2100, 0x1800, HUGE)],
        &[(0x1000, 0x100, META), (0x1100, 0x1000, META), (0x2100, 0x1800, META)],
    )]
    #[case::a_huge_object_of_a_page_or_more_apart_from_metadata_off_a_page_boundary_is_raw(
        &[(0x1000, 0x100, HEADER), (0x3100, 0x1800, HUGE)],
        &[(0x1000, 0x100, META), (0x3100, 0x1800, RAW)],
    )]
    fn fractal_heap_allocations_return_to_the_pages_they_occupy(
        #[case] allocations: &[(u64, u64, FractalHeapAllocation)],
        #[case] expected: &[(u64, u64, FreeClass)],
    ) {
        let page_size = FileSpacePageSize::try_from(0x1000).unwrap();

        assert_eq!(
            fractal_heap_free_spans(Some(page_size), allocations),
            expected
        );
    }

    #[test]
    fn without_pages_each_fractal_heap_allocation_takes_its_libhdf5_page_type() {
        let allocations = [
            (0x0f80, 0x80, HUGE),
            (0x1000, 0x100, HEADER),
            (0x1100, 0x80, HUGE),
        ];

        assert_eq!(
            fractal_heap_free_spans(None, &allocations),
            [
                (0x0f80, 0x80, RAW),
                (0x1000, 0x100, META),
                (0x1100, 0x80, RAW)
            ]
        );
    }

    #[rstest]
    #[case::without_pages(None, 0x0800, 0x1000, RAW)]
    #[case::below_a_page(Some(0x2000), 0x2000, 0x1000, RAW)]
    #[case::a_page_or_more_off_a_page_boundary(Some(0x1000), 0x1800, 0x1000, FreeClass::Dead)]
    #[case::a_page_or_more_on_a_page_boundary(Some(0x1000), 0x1000, 0x1000, FreeClass::Dead)]
    #[case::in_page_zero(Some(0x2000), 0x0800, 0x1000, FreeClass::Dead)]
    #[case::across_the_end_of_page_zero(Some(0x2000), 0x1f00, 0x1000, FreeClass::Dead)]
    fn a_collection_is_raw_data_below_a_page_and_outside_page_zero(
        #[case] page_size: Option<u64>,
        #[case] stored: u64,
        #[case] len: u64,
        #[case] expected: FreeClass,
    ) {
        let page_size = page_size.map(|size| FileSpacePageSize::try_from(size).unwrap());

        assert_eq!(collection_free_class(page_size, stored, len), expected);
    }

    const HEADER: FractalHeapAllocation = FractalHeapAllocation::Header;
    const DIRECT: FractalHeapAllocation = FractalHeapAllocation::DirectBlock;
    const INDEX: FractalHeapAllocation = FractalHeapAllocation::HugeObjectIndex;
    const HUGE: FractalHeapAllocation = FractalHeapAllocation::HugeObject;
    const META: FreeClass = FreeClass::Page(PageType::Meta);
    const RAW: FreeClass = FreeClass::Page(PageType::Raw);
}
