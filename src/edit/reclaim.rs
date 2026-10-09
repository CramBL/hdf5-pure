//! Plans storage reclamation for objects that lose their final owning hard link.
//!
//! This module proves object-header, dataset, dense-index, and subtree ownership before returning
//! physical spans to the editor. It also assigns PAGE metadata/raw classes where the on-disk
//! placement proves them. Any incomplete ownership proof leaks conservatively.

use std::collections::{HashMap, HashSet};

use hdf5_pure_space::__private::{FreeClass, PageType};

use crate::access_mode::AccessMode;
use crate::address::{BaseAddressExt, StoredAddress};
use crate::attribute_info::AttributeInfoMessage;
use crate::chunked_read::chunk_index_spans_from_source;
use crate::data_layout::{ChunkIndexLayout, DataLayout};
use crate::dataspace::Dataspace;
use crate::error::Error;
use crate::file_space_info::FileSpacePageSize;
use crate::group_v2::resolve_group_entries_from_source;
use crate::link_info::LinkInfoMessage;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::source::{BaseOffsetSource, Source, SourceMetadata};

use super::{
    LENGTH_SIZE, MAX_COPY_DEPTH, MAX_LINK_GRAPH_NODES, OFFSET_SIZE, ObjModel, WriteEngine,
    read_oh_chunks, spans_disjoint_in_bounds,
};

impl WriteEngine {
    /// Returns the on-disk byte spans `(addr, len)` of every chunk of the
    /// version 2 object header at `addr`: chunk 0 (signature, prefix, messages,
    /// checksum) plus each continuation (`OCHK`) block.
    ///
    /// Used to reclaim a header's storage when its object is deleted. An error
    /// (propagated from [`super::oh_region_at`] or a malformed continuation) means the
    /// header is not a plain v2 header this engine can fully account for, and
    /// the caller leaves it as dead bytes of unknown extent.
    pub(super) fn oh_chunk_spans(&self, addr: u64) -> Result<Vec<(u64, u64)>, Error> {
        Ok(
            read_oh_chunks(&self.image(), addr, self.superblock.base_address)?
                .into_iter()
                .map(|chunk| chunk.span)
                .collect(),
        )
    }

    /// Returns the dense-attribute v2 B-tree allocations named by an object's old header.
    ///
    /// The Attribute Info message separates the fractal heap from its name and optional
    /// creation-order indexes. This returns only the index allocations and leaves the heap in
    /// place. Both indexes are proven together, then checked against the old object-header
    /// allocations before any span is returned. Replacement headers cannot overlap these extents:
    /// the commit gathers old storage in `to_free`, places every replacement, and only then admits
    /// `to_free` to the session space. `None` leaves the index set unreclaimed.
    pub(super) fn dense_attribute_index_spans(
        &self,
        addr: u64,
        header_spans: &[(u64, u64)],
    ) -> Option<Vec<(u64, u64)>> {
        let base = self.superblock.base_address;
        let region = Self::gather_oh_messages(&self.image(), addr, base).ok()?;
        let mut attr_info = None;
        let mut p = 0;
        while let Some((msg_type, body, body_end)) = region.next_message(p).ok()? {
            if msg_type == MessageType::ATTRIBUTE_INFO {
                if attr_info.is_some() {
                    return None;
                }
                attr_info =
                    Some(AttributeInfoMessage::parse(&region[body..body_end], OFFSET_SIZE).ok()?);
            }
            p = body_end;
        }
        let Some(attr_info) = attr_info else {
            return Some(Vec::new());
        };
        if attr_info.fractal_heap_address.is_none() {
            return Some(Vec::new());
        }

        let extents = crate::attribute::collect_dense_attribute_index_storage_extents(
            &BaseOffsetSource {
                inner: &self.image(),
                base,
            },
            &attr_info,
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .ok()?;
        let mut index_spans = Vec::with_capacity(extents.len());
        for extent in extents {
            index_spans.push((
                base.absolute(StoredAddress::new(extent.start())).ok()?,
                extent.len(),
            ));
        }

        let mut owned = header_spans.to_vec();
        owned.extend(index_spans.iter().copied());
        spans_disjoint_in_bounds(&mut owned, self.image.len()).then_some(index_spans)
    }

    /// Returns the dense-group version 2 B-tree allocations identified by an old group header.
    ///
    /// The Link Info message separates the fractal heap from its type 5 name index and optional
    /// type 6 creation-order index. Only the indexes are returned. The complete index set is
    /// proven together and checked against the old object-header allocations before any span
    /// joins the deferred reclamation plan. `None` leaves the index set unreclaimed.
    fn dense_group_index_spans(
        &self,
        addr: u64,
        header_spans: &[(u64, u64)],
    ) -> Option<Vec<(u64, u64)>> {
        let base = self.superblock.base_address;
        let region = Self::gather_oh_messages(&self.image(), addr, base).ok()?;
        let mut link_info = None;
        let mut p = 0;
        while let Some((msg_type, body, body_end)) = region.next_message(p).ok()? {
            if msg_type == MessageType::LINK_INFO {
                if link_info.is_some() {
                    return None;
                }
                link_info =
                    Some(LinkInfoMessage::parse(&region[body..body_end], OFFSET_SIZE).ok()?);
            }
            p = body_end;
        }
        let Some(link_info) = link_info else {
            return Some(Vec::new());
        };
        if link_info.fractal_heap_address.is_none() {
            return Some(Vec::new());
        }

        let extents = crate::group_v2::collect_dense_group_index_storage_extents(
            &BaseOffsetSource {
                inner: &self.image(),
                base,
            },
            &link_info,
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .ok()?;
        let mut index_spans = Vec::with_capacity(extents.len());
        for extent in extents {
            index_spans.push((
                base.absolute(StoredAddress::new(extent.start())).ok()?,
                extent.len(),
            ));
        }

        let mut owned = header_spans.to_vec();
        owned.extend(index_spans.iter().copied());
        spans_disjoint_in_bounds(&mut owned, self.image.len()).then_some(index_spans)
    }

    /// Count, for every object-header address reachable from the root, how many
    /// hard links in the *pre-commit* file point to it. The result drives the
    /// last-hard-link reclaim guard in [`collect_free_spans`](Self::collect_free_spans):
    /// an object is freed only when its count is 1.
    ///
    /// Walks the whole link graph from the root, following hard links through
    /// groups of any on-disk format (v0/v1 symbol-table, v2 compact, v2 dense)
    /// via [`resolve_group_entries_from_source`], tallying each hard-link edge. Datasets and
    /// other leaves contribute no edges. Returns `None` if the graph cannot be walked in full:
    /// an unparseable header, a group whose links cannot be enumerated, or
    /// more than [`MAX_LINK_GRAPH_NODES`] objects. Cycles are handled by visiting
    /// each object once. Base-aware: stored child addresses are shifted by the
    /// userblock base, so the returned keys are absolute file offsets.
    pub(super) fn count_incoming_hard_links(&self) -> Option<HashMap<u64, u32>> {
        let os = self.superblock.offset_size;
        let ls = self.superblock.length_size;
        let base = self.superblock.base_address;
        let mut counts: HashMap<u64, u32> = HashMap::new();
        let mut visited: HashSet<u64> = HashSet::new();
        let mut stack: Vec<u64> = vec![self.superblock.root_group_address];
        let mut budget = MAX_LINK_GRAPH_NODES;
        while let Some(addr) = stack.pop() {
            if !visited.insert(addr) {
                continue; // already expanded (also breaks hard-link cycles)
            }
            if budget == 0 {
                return None; // graph exceeds the walk limit, so leak conservatively
            }
            budget -= 1;
            let header = ObjectHeader::parse_from_source(
                &SourceMetadata(&self.image()),
                AccessMode::ReadWrite,
                addr,
                os,
                ls,
                base,
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
            let entries =
                resolve_group_entries_from_source(&self.image(), &header, os, ls, base).ok()?;
            for e in entries {
                let child = base.absolute(e.object_header_address).ok()?;
                *counts.entry(child).or_insert(0) += 1;
                stack.push(child);
            }
        }
        Some(counts)
    }

    /// Collects, as far as it can account for them, the span and free class of
    /// every on-disk block the object at `addr` owns, and for a group its whole
    /// subtree, into `out` for reclamation after a delete.
    ///
    /// Contiguous datasets (header + data block), chunked datasets (header +
    /// chunk index + chunk data, via [`chunked_storage_spans`](Self::chunked_storage_spans)),
    /// and whole group subtrees are reclaimed. Deliberately conservative: any
    /// object whose layout it cannot fully account for, such as a non-v2 header, an
    /// unsupported or only-partially-enumerable chunk index, or a group holding a soft/external
    /// link contributes nothing and is not descended into, so `out` never contains a region that
    /// might still be in use. Dense attribute and dense group indexes are added only after each
    /// complete index set is proven. Dense group index ownership is separate from ownership of the
    /// child objects its links reach. The associated fractal heaps stay in place. Bounded by
    /// [`MAX_COPY_DEPTH`] against a hard-link cycle.
    /// Variable-length data in global-heap collections is never reclaimed here (a
    /// collection can be shared between objects), so it is simply left behind.
    ///
    /// `incoming` is the file-wide hard-link count per object-header address
    /// (from [`count_incoming_hard_links`](Self::count_incoming_hard_links)). An
    /// object is reclaimed, and a group is descended into, only when its
    /// count is exactly 1, i.e. the link being removed is its last: an object
    /// still reachable through another hard link is live and is left untouched
    /// (so is everything below a surviving group), which is what keeps deleting
    /// one of several hard links from corrupting the survivor.
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
        let base = self.superblock.base_address;
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
        let dense_index_spans = self.dense_attribute_index_spans(addr, &spans);
        match Self::read_object(
            &self.image(),
            AccessMode::ReadWrite,
            addr,
            self.superblock.base_address,
        ) {
            Ok(ObjModel::DatasetVerbatim { .. }) => {
                append_object_metadata_spans(out, &spans, dense_index_spans.as_deref());
            }
            Ok(ObjModel::DatasetContiguous {
                data_addr,
                data_size,
                ..
            }) => {
                append_object_metadata_spans(out, &spans, dense_index_spans.as_deref());
                // A defined, in-bounds contiguous data block is owned outright.
                // An empty dataset stores the undefined address and owns none. Its
                // absolute file offset is what this bounds-checks and records.
                if !data_addr.is_undefined(OFFSET_SIZE) && data_size > 0 {
                    if let Ok(abs) = base.absolute(data_addr) {
                        if abs.checked_add(data_size).is_some_and(|e| e <= file_len) {
                            // A contiguous data block is raw data.
                            out.push((abs, data_size, FreeClass::Page(PageType::Raw)));
                        }
                    }
                }
            }
            Ok(ObjModel::Group { children, .. }) => {
                let dense_group_index_spans = self.dense_group_index_spans(addr, &spans);
                append_object_metadata_spans(out, &spans, dense_index_spans.as_deref());
                if let Some(index_spans) = dense_group_index_spans {
                    // Dense-link B-tree headers and nodes are file metadata under PAGE.
                    out.extend(meta_spans(index_spans));
                }
                // The recursion works in absolute offsets, matching `incoming`'s
                // keys and `oh_chunk_spans`.
                for (_, _, child) in children {
                    if let Ok(c) = base.absolute(child) {
                        self.collect_free_spans(c, depth + 1, incoming, out);
                    }
                }
            }
            // A chunked dataset: reclaim its chunk index and chunk data blocks
            // alongside its header. `chunked_storage_spans` returns `None` for
            // anything it cannot account for exhaustively (an index type with no
            // walker, an undefined index address, or spans that fail the
            // bounds/overlap check), leaving the whole dataset as dead bytes and not freeing a
            // region that might still be in use.
            Ok(ObjModel::DatasetChunked { .. }) => {
                if let Some(storage) = self.chunked_storage_spans(addr) {
                    append_object_metadata_spans(out, &spans, dense_index_spans.as_deref());
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
    /// [`chunked_read::collect_chunked_storage_spans`](crate::chunked_read::collect_chunked_storage_spans).
    /// This method only locates the layout and dataspace messages and validates
    /// the result. Variable-length data in global-heap collections is still
    /// never reclaimed (a collection can be shared between objects). See the
    /// [module docs](self).
    pub(super) fn chunked_storage_spans(&self, addr: u64) -> Option<Vec<(u64, u64, FreeClass)>> {
        // Locate the data-layout and dataspace messages in the object header.
        let region =
            Self::gather_oh_messages(&self.image(), addr, self.superblock.base_address).ok()?;
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

        let layout = DataLayout::parse(&region[lb..le], OFFSET_SIZE, LENGTH_SIZE).ok()?;
        if !matches!(layout, DataLayout::Chunked { .. }) {
            return None;
        }
        let dataspace = Dataspace::parse(&region[db..de], LENGTH_SIZE).ok()?;

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
        let base = self.superblock.base_address;
        let split = crate::chunked_read::collect_chunked_storage_spans(
            &BaseOffsetSource {
                inner: &self.image(),
                base,
            },
            &layout,
            &dataspace,
            OFFSET_SIZE,
            LENGTH_SIZE,
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
        if !spans_disjoint_in_bounds(&mut plain, self.image.len()) {
            return None;
        }
        let chunk_index = match layout {
            DataLayout::Chunked { index, .. } => index,
            _ => return None,
        };
        let index_class =
            chunk_index_free_class(chunk_index, self.index_is_provably_raw(&data, &index));
        let mut spans: Vec<(u64, u64, FreeClass)> = data
            .iter()
            .map(|&(a, l)| (a, l, FreeClass::Page(PageType::Raw)))
            .collect();
        spans.extend(index.into_iter().map(|(a, l)| (a, l, index_class)));
        Some(spans)
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
                index_abuts_chunk_data(data, index) && !index_touches_page_zero(index, page_size)
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
    /// [`chunk_index_spans_from_source`], which enumerates the index structure's
    /// own blocks (B-tree v1/v2 nodes, or fixed- / extensible-array header, index,
    /// super, and data blocks) and never a chunk-data address, so the shared
    /// kept chunk data is never freed. Base-aware and validated
    /// disjoint/in-bounds. Returns `None` (leave unreclaimed) on any error or
    /// violation.
    pub(super) fn chunked_index_spans(&self, addr: u64) -> Option<Vec<(u64, u64)>> {
        let region =
            Self::gather_oh_messages(&self.image(), addr, self.superblock.base_address).ok()?;
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
        let layout = DataLayout::parse(&region[lb..le], OFFSET_SIZE, LENGTH_SIZE).ok()?;
        if !matches!(layout, DataLayout::Chunked { .. }) {
            return None;
        }
        let base = self.superblock.base_address;
        let mut spans = chunk_index_spans_from_source(
            &BaseOffsetSource {
                inner: &self.image(),
                base,
            },
            &layout,
            OFFSET_SIZE,
            LENGTH_SIZE,
        )
        .ok()?;
        for (a, _) in &mut spans {
            *a = base.absolute(StoredAddress::new(*a)).ok()?;
        }
        if !spans_disjoint_in_bounds(&mut spans, self.image.len()) {
            return None;
        }
        Some(spans)
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

/// Whether any byte of the `index` run lies in page 0 of a paged file, whose
/// first bytes are the superblock and which is therefore a metadata page. The
/// screening half of [`WriteEngine::index_is_provably_raw`], where the reasoning
/// lives.
pub(super) fn index_touches_page_zero(index: &[(u64, u64)], page_size: FileSpacePageSize) -> bool {
    index
        .iter()
        .any(|&(addr, len)| len > 0 && addr < page_size.get())
}

/// Adds object-header metadata and any independently proven dense-attribute indexes.
fn append_object_metadata_spans(
    out: &mut Vec<(u64, u64, FreeClass)>,
    header_spans: &[(u64, u64)],
    dense_index_spans: Option<&[(u64, u64)]>,
) {
    out.extend(
        header_spans
            .iter()
            .chain(dense_index_spans.into_iter().flatten())
            .map(|&(addr, len)| (addr, len, FreeClass::Page(PageType::Meta))),
    );
}

/// Tags object-header chunk spans as file metadata.
///
/// Every span [`WriteEngine::oh_chunk_spans`] returns is part of an object header, so the page type
/// is the same for all of them.
pub(super) fn meta_spans(spans: Vec<(u64, u64)>) -> impl Iterator<Item = (u64, u64, FreeClass)> {
    spans
        .into_iter()
        .map(|(a, l)| (a, l, FreeClass::Page(PageType::Meta)))
}
