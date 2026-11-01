//! Repointing the object references a file already stores, across the
//! object-header relocations one commit makes (issue #324).
//!
//! A [`File::open_rw`](crate::File::open_rw) commit rebuilds a dirty object's
//! header at a fresh address and repoints the one *link* that names it. That is
//! enough for the link graph and not enough for the file: an **object
//! reference** is an object-header address stored as data, in a dataset's
//! elements or an attribute's value, and nothing about rewriting a header
//! updates one. Left alone it keeps naming the pre-commit header, which the same
//! commit frees — so a reference dataset the commit never touched reads a stale
//! copy of the object, and reads garbage once something reuses the span.
//!
//! This module finds those stored addresses and rewrites them. It runs *after*
//! the superblock repoint, over the committed tree, for two reasons. The commit
//! is atomic at the repoint, and a fixup derived from it must not precede it: a
//! crash before the repoint has to leave the pre-commit file, references
//! included. And the committed tree is the one whose bytes need fixing — a
//! rebuilt group's header is a fresh copy of the pre-commit one, carrying the
//! old address of anything it referenced, so patching the header the commit
//! superseded would correct bytes that are already dead.
//!
//! # What it reaches
//!
//! Two questions decide it, and both have to answer yes: **where** the bytes are
//! stored, and **what** [`datatype::embedded_reference_slots`] can locate inside an
//! element.
//!
//! Storage: an **inline attribute's value**, and a **contiguous** or **compact**
//! dataset's elements. A **committed** (`H5Tcommit`) datatype is followed rather
//! than skipped, so a dataset of references through a named type is reached like
//! any other.
//!
//! Datatype: an 8-byte object reference, an **array** of them, a **compound**
//! holding one, and any nesting of those two. That is the whole of it — see
//! `datatype::embedded_reference_slots`, whose reach this inherits exactly.
//!
//! # What it does not
//!
//! Everything else, and none of it is refused. Each is left as it was, which is
//! the pre-#324 behaviour rather than a new failure. By storage:
//!
//! - **Chunked** dataset elements. Unfiltered chunks would only need the index
//!   walked; filtered ones hold their addresses compressed, the same obstacle
//!   that makes [`crate::repack()`] reject a chunked object-reference dataset.
//! - **Dense (fractal-heap) attributes**, whose values are not in the header,
//!   and an attribute held as a **shared (SOHM) record**, which is not stored in
//!   the header either.
//! - An **attribute's value** in a **version 1 object header**.
//!   [`edit::object_header::read_oh_chunks`] supports version 2 headers.
//!   [`scan_parsed_header`] reads contiguous element data from parsed headers of both versions,
//!   whose message bodies have no file offsets for attribute patches. The walk descends through
//!   either version to find child objects.
//!
//! And by datatype, every form `embedded_reference_slots` answers `None` for: an
//! object reference **wider than 8 bytes**, a **dataset-region** reference, a
//! **variable length** of references (the encoding the dimension-scale attribute
//! `DIMENSION_LIST` uses, whose addresses sit in a global heap collection), an
//! **enumeration** over one, and any compound or array reaching one of those.
//!
//! A file whose objects all carry version 1 headers - libhdf5's
//! default, and h5py's — can never be *proved* free of references, since an
//! attribute in one of those headers is never read. Such a file is therefore
//! walked on every commit rather than on the first.
//!
//! Deliberately, none of these is a *refusal*. Every commit rebuilds at least
//! its root group, so there is always a relocation in hand, and refusing on an
//! unreachable shape would therefore refuse *every* commit on any file holding
//! one — a ban rather than a check, imposed on files whose references were no
//! better served before this module existed. The crate's
//! [limitations](crate#limitations) record the set as a whole.

use std::collections::BTreeMap;

use hdf5_pure_core::__private::FileRange;

use crate::PhysicalAddress;
use crate::access_mode::AccessMode;
#[cfg(test)]
use crate::address::BaseAddress;
#[cfg(test)]
use crate::address::BaseAddressExt;
use crate::address::StoredAddressExt;
use crate::address::{AddressContext, StoredAddress};
use crate::attribute::AttributeMessage;
use crate::checksum;
use crate::convert::Narrow;
use crate::data_layout::{COMPACT_DATA_OFFSET, DataLayout};
use crate::datatype::{self, Datatype};
use crate::edit;
use crate::error::Error;
use crate::group_v1;
use crate::group_v2;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_graph::{ObjectId, ObjectTraversal, TraversalBudget};
use crate::object_header::ObjectHeader;
use crate::source::OriginalSource;
use crate::source::SourceMetadata;
use crate::superblock::Superblock;
use crate::width::FormatWidths;

/// The byte edits that repoint a commit's relocated object headers.
///
/// Built whole before anything is written, and split by where the bytes live,
/// because that is what decides how they may be written: an edit inside an
/// object header has to travel with the checksum covering it, and one in a data
/// block stands alone.
#[derive(Debug, Default)]
pub(crate) struct Plan {
    /// Edits in a dataset's data block: `(absolute file offset, the
    /// base-relative address to store there)`. A data block carries no checksum,
    /// so each of these is an independent eight-byte publish. A crash between
    /// two of them leaves some elements repointed and some not — a mixture of
    /// the pre-commit address and the post-commit one, which is the state every
    /// such element was left in before this existed.
    data_writes: Vec<(u64, StoredAddress)>,
    /// Edits within version 2 header chunks, grouped by complete physical range.
    ///
    /// Each `(offset, value)` uses a byte displacement from the chunk's start. [`Plan::apply`]
    /// reads the complete chunk, recalculates its checksum, and writes the changed suffix and
    /// checksum together.
    header_writes: BTreeMap<FileRange, Vec<(usize, StoredAddress)>>,
    /// Whether the walk reached every object in the file and found none whose
    /// datatype holds an object address — the proof that no later commit on this
    /// session need walk it again, provided nothing is added meanwhile.
    ///
    /// False whenever the answer is not established, which covers three
    /// different situations deliberately folded together: a reference *was*
    /// found, the walk was cut short by its budget, and the walk did not run.
    /// Only the first of those is common, and all three mean the same thing to
    /// the caller.
    proved_free_of_references: bool,
}

/// The file a [`Plan`] is applied to. Readable as well as writable, because an
/// object header is republished as a whole checksummed chunk: the bytes around
/// an edit have to be read back before the checksum over them can be
/// recomputed.
pub(crate) trait PatchTarget {
    fn read(&self, at: u64, len: usize) -> Result<Vec<u8>, Error>;
    fn write(&mut self, at: u64, bytes: &[u8]) -> Result<(), Error>;
}

impl Plan {
    /// Whether the plan would write anything.
    pub(crate) fn is_empty(&self) -> bool {
        self.data_writes.is_empty() && self.header_writes.is_empty()
    }

    /// Whether this walk proved the file holds no object reference at all. See
    /// the field of the same name.
    pub(crate) fn proved_free_of_references(&self) -> bool {
        self.proved_free_of_references
    }

    /// Writes the planned object references and corresponding header checksums.
    ///
    /// Reads each complete header chunk before updating its elements and checksum, then writes
    /// from the first edited byte through the checksum. Element updates in dataset data blocks
    /// are separate eight-byte writes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a header chunk is shorter than its checksum or an
    /// edit extends outside its message bytes. Returns [`Error::Format`] if a chunk's length
    /// exceeds [`usize::MAX`]. Propagates patch-target read and write errors.
    pub(crate) fn apply(&self, target: &mut impl PatchTarget) -> Result<(), Error> {
        for &(at, value) in &self.data_writes {
            target.write(at, &value.get().to_le_bytes())?;
        }
        for (&range, edits) in &self.header_writes {
            let at = range.start().get();
            let len = range.length();
            // `read_oh_chunks` produced this span from a header it had already
            // bounds-checked, and the length always covers the trailing
            // four-byte checksum.
            let len = len.to_usize()?;
            let body_len = len
                .checked_sub(4)
                .ok_or(Error::EditUnsupported("object header chunk is too short"))?;
            let mut chunk = target.read(at, len)?;
            let mut from = body_len;
            for &(offset, value) in edits {
                // Refused rather than asserted away: every offset here is derived
                // from geometry parsed out of the file, so one that leaves the
                // chunk is untrusted input rather than a broken invariant.
                let end = offset.checked_add(8).filter(|&e| e <= body_len).ok_or(
                    Error::EditUnsupported(
                        "a stored reference sits outside the object header chunk holding it",
                    ),
                )?;
                chunk[offset..end].copy_from_slice(&value.get().to_le_bytes());
                from = from.min(offset);
            }
            let checksum = checksum::jenkins_lookup3(&chunk[..body_len]);
            chunk[body_len..].copy_from_slice(&checksum.to_le_bytes());
            target.write(at + from as u64, &chunk[from..])?;
        }
        Ok(())
    }

    /// How many stored references the plan repoints. Reported by the session's
    /// own tests rather than by any public API.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.data_writes.len() + self.header_writes.values().map(Vec::len).sum::<usize>()
    }

    /// The addresses this plan rewrites, for tests that pin *where* it acts.
    #[cfg(test)]
    pub(crate) fn sites(&self) -> Vec<u64> {
        let headers = self.header_writes.iter().flat_map(|(&range, edits)| {
            edits
                .iter()
                .map(move |&(o, _)| range.start().checked_add(o as u64).unwrap().get())
        });
        self.data_writes
            .iter()
            .map(|&(at, _)| at)
            .chain(headers)
            .collect()
    }
}

std::thread_local! {
    /// How many times a commit on this thread has walked the file to repoint its
    /// stored references.
    ///
    /// The walk's *absence* is the whole of the `proved_free_of_references`
    /// optimization, and absence is invisible from outside: a session that never
    /// caches the proof produces byte-identical files and passes every
    /// correctness test, having quietly gone back to walking the file on every
    /// commit. The count is the only thing a test can hold that to. Per-thread
    /// because the harness runs tests concurrently, each on its own thread.
    static WALKS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// Start counting reference walks on this thread from zero.
#[cfg(test)]
pub(crate) fn reset_walks() {
    WALKS.with(|count| count.set(0));
}

/// How many reference walks this thread has run since the last reset.
#[cfg(test)]
pub(crate) fn walks() -> usize {
    WALKS.with(|count| count.get())
}

/// Plan the repointing for `relocations` against the file `src` currently holds.
///
/// `relocations` maps each vacated object-header address to the address the same
/// commit rewrote it to, both absolute. `superblock` is the *committed* one, so
/// its root is the tree to walk.
///
/// The walk is bounded by `budget` objects and terminates on cycles (hard links
/// can form them). Running out of budget is not an error: the plan keeps and
/// applies what it reached, and only the *proof* of a reference-free file is
/// withheld. That is the opposite of what `WriteEngine::count_incoming_hard_links`
/// does with a graph too large to walk — it discards the whole walk and reclaims
/// nothing — and the two differ because their partial results do. Half a link
/// count would over-reclaim and corrupt a survivor; half a repointing is half
/// the references corrected and the rest exactly as they were.
pub(crate) fn plan<S: OriginalSource + ?Sized>(
    src: &S,
    superblock: &Superblock,
    widths: FormatWidths,
    relocations: &BTreeMap<u64, u64>,
    budget: u32,
) -> Result<Plan, Error> {
    if relocations.is_empty() {
        return Ok(Plan::default());
    }
    walk(src, superblock, widths, relocations, budget)
}

/// Returns whether every reachable object was inspected and has no object references.
///
/// Returns `false` if traversal exceeds `budget`, encounters unsupported storage, or cannot
/// inspect every object.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] for a truncated object-header record and [`Error::Format`]
/// for other record errors or a root-address overflow.
pub(crate) fn prove_free_of_references(
    src: &(impl OriginalSource + ?Sized),
    superblock: &Superblock,
    widths: FormatWidths,
    budget: u32,
) -> Result<bool, Error> {
    Ok(walk(src, superblock, widths, &BTreeMap::new(), budget)?.proved_free_of_references())
}

fn walk(
    src: &(impl OriginalSource + ?Sized),
    superblock: &Superblock,
    widths: FormatWidths,
    relocations: &BTreeMap<u64, u64>,
    budget: u32,
) -> Result<Plan, Error> {
    let mut plan = Plan::default();
    WALKS.with(|count| count.set(count.get() + 1));
    let addresses = src.addresses();
    let base = addresses;

    let mut traversal = ObjectTraversal::new(
        [ObjectId::resolve(addresses, superblock.root_group_address)?],
        TraversalBudget::new(budget as usize, usize::MAX),
        u32::MAX,
    );
    let mut saw_reference = false;
    let mut complete = true;
    while let Some((id, depth)) =
        traversal.next_matching(|id| !relocations.contains_key(&id.header_address().get()))
    {
        let addr = id.header_address().get();
        // One pass over the header serves both needs: it collects the byte edits
        // this object owes, and it reports whether the object is a group. Only a
        // group is parsed a second time, into the form
        // `resolve_group_entries_from_source` needs. The file's datasets are the
        // many, and parsing every one of them again only to learn it has no
        // children was most of what this walk cost. Measured on a release build
        // over a file of 8,000 root datasets, timing `plan` alone across five
        // commits: 32.3 ms a commit when every object was parsed twice, 13.4 ms
        // once only groups were. No test holds those figures — unlike
        // `publish_checksummed`, whose claim its own crash test pins — so treat
        // them as a record of one measurement rather than as a bound.
        let outcome = scan_object(src, addr, widths, relocations, &mut plan)?;
        saw_reference |= outcome.holds_a_reference;
        complete &= outcome.fully_read;
        if !outcome.descend {
            continue;
        }
        // A group whose header or links cannot be read hides its subtree from
        // the walk; that leaves references below it unpatched, never mispatched.
        let Ok(header) = ObjectHeader::parse_from_source_with_widths(
            &SourceMetadata(src),
            AccessMode::ReadWrite,
            addr,
            widths,
            base.effective_base(),
        ) else {
            complete = false;
            continue;
        };
        // A header the chunk reader declined is reached through this parse
        // instead, which reads version 1 and creation-order headers alike. It
        // gives message *bodies* rather than their file offsets, so it reaches
        // element data — which lives outside the header — and not an attribute's
        // value, which does not.
        if !outcome.header_located {
            saw_reference |= scan_parsed_header(src, &header, widths, relocations, &mut plan);
        }
        let Ok(entries) = group_v2::resolve_group_entries_from_source(
            src,
            AccessMode::ReadWrite,
            &header,
            widths,
            group_v1::group_leaf_node_k(superblock),
        ) else {
            complete = false;
            continue;
        };
        for e in entries {
            if let Ok(child) = base
                .resolve(e.object_header_address)
                .map(PhysicalAddress::get)
            {
                traversal.push_child(ObjectId::new(PhysicalAddress::new(child)), depth);
            }
        }
    }
    plan.proved_free_of_references = complete && traversal.complete() && !saw_reference;
    Ok(plan)
}

/// Collects reference patches in contiguous data using a parsed object header.
///
/// The parsed messages have no file offsets for inline attribute or compact data patches.
/// Returns whether the element datatype can contain an object reference.
fn scan_parsed_header<S: OriginalSource + ?Sized>(
    src: &S,
    header: &ObjectHeader,
    widths: FormatWidths,
    relocations: &BTreeMap<u64, u64>,
    plan: &mut Plan,
) -> bool {
    use crate::shared_message::SharedResolver as _;
    let base = src.addresses();
    let mut element_dt = None;
    let mut layout = None;
    for message in &header.messages {
        match message.msg_type {
            // A committed (shared) datatype's body names the type rather than
            // encoding it, so it is followed first — the flag is on the message,
            // which this form does keep. Reading the class byte off an
            // unresolved body would be reading a shared-message header as a
            // datatype class, which is not an error, just an answer about the
            // wrong bytes.
            MessageType::DATATYPE => {
                let resolved;
                let encoded = if message.flags.is_shared() {
                    let resolver = crate::shared_message::SourceResolver::new(
                        src,
                        AccessMode::ReadWrite,
                        widths,
                        // No shared-message table: a heap-stored datatype is not
                        // followed here, and the arm below treats an unresolved
                        // one as a datatype this walk cannot read.
                        None,
                    );
                    match resolver.resolve(&message.data, MessageType::DATATYPE) {
                        Ok(bytes) => {
                            resolved = bytes;
                            &resolved[..]
                        }
                        Err(_) => continue,
                    }
                } else {
                    &message.data[..]
                };
                if encoded
                    .first()
                    .is_some_and(|&b| crate::datatype::class_may_hold_object_address(b))
                {
                    if let Ok((dt, _)) = hdf5_pure_format::__private::parse_datatype(encoded) {
                        element_dt = Some(dt);
                    }
                }
            }
            MessageType::DATA_LAYOUT => {
                layout = DataLayout::parse_with_widths(&message.data, widths).ok();
            }
            _ => {}
        }
    }
    let Some(dt) = element_dt else {
        return false;
    };
    let holds = datatype::datatype_holds_object_address(&dt);
    let (
        Some(slots),
        Some(DataLayout::Contiguous {
            address: Some(a),
            size,
        }),
    ) = (element_slots(&dt), layout)
    else {
        return holds;
    };
    let (Ok(at), Ok(want)) = (
        base.resolve(a).map(PhysicalAddress::get),
        usize::try_from(size),
    ) else {
        return holds;
    };
    let Ok(raw) = src.read_exact_at(at, want) else {
        return holds;
    };
    collect_slots(
        &dt,
        &slots,
        &raw,
        at,
        base,
        relocations,
        &mut plan.data_writes,
    );
    holds
}

/// What one object contributed to the walk beyond its byte edits.
struct Scanned {
    /// Whether the caller should look for children below this object.
    ///
    /// True if the header contains group messages or the chunk traversal failed. The caller
    /// can attempt the parsed-header traversal after a chunk traversal error.
    descend: bool,
    /// Some datatype in the object — its elements' or an attribute's — reaches
    /// an object address. True whether or not any address needed repointing:
    /// the question this answers is whether the file *can* hold one, which is
    /// what lets a later commit skip the walk.
    holds_a_reference: bool,
    /// The object's header was read in full. False leaves the file's
    /// reference-free claim unproven, since what was not read might hold one.
    fully_read: bool,
    /// Whether the chunk traversal located the messages for patches within the header.
    ///
    /// False for a version 1 header, whose contiguous data is read by [`scan_parsed_header`].
    header_located: bool,
}

/// Collect the byte edits owed by the single object whose header is at `addr`:
/// its inline attributes' values, and its own elements when they are stored
/// where this can address them.
///
/// Reports whether the caller should descend through this object looking for
/// children, which this walk establishes on the way past — see
/// [`Scanned::descend`], which is not quite "this object is a group".
fn scan_object<S: OriginalSource + ?Sized>(
    src: &S,
    addr: u64,
    widths: FormatWidths,
    relocations: &BTreeMap<u64, u64>,
    plan: &mut Plan,
) -> Result<Scanned, Error> {
    // A header this cannot read — a version 1 one — contributes no edits, and
    // is reported as a group so the caller still tries to descend through it:
    // `ObjectHeader::parse_from_source` reads both, so such a group does not
    // hide the objects below it from the walk.
    let base = src.addresses();
    let Ok(chunks) = edit::object_header::read_oh_chunks(src, addr, base) else {
        return Ok(Scanned {
            descend: true,
            holds_a_reference: false,
            fully_read: false,
            header_located: false,
        });
    };
    use crate::shared_message::SharedResolver as _;
    // Resolve committed datatype addresses through the opened source's effective base.
    let resolver = crate::shared_message::SourceResolver::new(
        src,
        AccessMode::ReadWrite,
        widths,
        // No shared-message table; see the note on the other resolver in this
        // module.
        None,
    );

    let mut element_dt: Option<Datatype> = None;
    // The data-layout message's body and its absolute offset, kept unparsed:
    // where it points only matters once the datatype says the elements could
    // hold an address, and most datasets say they could not.
    let mut layout_msg: Option<(&[u8], u64)> = None;
    let mut out = Scanned {
        descend: false,
        holds_a_reference: false,
        fully_read: true,
        header_located: true,
    };

    // Resolved committed-datatype bodies, kept alive for the borrows the message
    // walk takes of them. One per header; a dataset has at most one datatype.
    let mut committed: Option<Vec<u8>> = None;
    // Reused across messages: one attribute's edits at a time.
    let mut edits: Vec<(u64, StoredAddress)> = Vec::new();
    for chunk in &chunks {
        let layout = chunk.layout();
        let (region, mut p) = chunk.message_region();
        while let Some((msg_type, body, body_end)) =
            crate::edit::object_header::next_record(layout, region, p)?
        {
            // Absolute file offset of this message's body.
            let body_at = chunk.span.start().get() + body as u64;
            // The flags byte is the 4th of the record header (type, size, flags).
            let shared = MessageFlags::new(region[p + 3]).is_shared();
            match msg_type {
                MessageType::SYMBOL_TABLE | MessageType::LINK | MessageType::LINK_INFO => {
                    out.descend = true;
                }
                MessageType::DATATYPE => {
                    // A committed (shared) datatype's message body names the
                    // type rather than encoding it, so it is followed before the
                    // class can be read at all. A dataset of references through
                    // a committed type is otherwise invisible here, elements and
                    // all.
                    let encoded = if shared {
                        match resolver.resolve(&region[body..body_end], MessageType::DATATYPE) {
                            Ok(bytes) => committed.insert(bytes),
                            Err(_) => {
                                out.fully_read = false;
                                p = body_end;
                                continue;
                            }
                        }
                    } else {
                        &region[body..body_end]
                    };
                    // Parsing a datatype allocates, and this walk sees every
                    // dataset in the file on every commit, so the encoded class
                    // decides first: it is a necessary condition for holding an
                    // address and costs one byte to read.
                    if encoded.is_empty() || !datatype::class_may_hold_object_address(encoded[0]) {
                        p = body_end;
                        continue;
                    }
                    match hdf5_pure_format::__private::parse_datatype(encoded) {
                        Ok((dt, _)) => {
                            out.holds_a_reference |= datatype::datatype_holds_object_address(&dt);
                            element_dt = Some(dt);
                        }
                        // A datatype of a qualifying class that will not parse
                        // might have been a reference; the file is not proven
                        // free of them.
                        Err(_) => out.fully_read = false,
                    }
                }
                MessageType::DATA_LAYOUT => {
                    layout_msg = Some((&region[body..body_end], body_at));
                }
                // Dense (fractal-heap) attributes are not held in the header at
                // all, so an object storing them this way presents no Attribute
                // message here and would otherwise read as *proven* free of
                // references while holding any number of them. The test is a
                // defined heap address rather than the message's presence: the
                // reference C library and h5py emit an Attribute Info message
                // for compact attributes too, to carry creation-order metadata.
                MessageType::ATTRIBUTE_INFO
                    if crate::edit::attributes::attribute_info_is_dense(
                        &region[body..body_end],
                    ) =>
                {
                    out.fully_read = false;
                }
                // A *shared record* attribute — the whole message held in the
                // file's shared-message table — is not stored here, so its
                // elements are not addressable from this header, and it leaves
                // the object unproven rather than proven reference-free.
                MessageType::ATTRIBUTE if shared => out.fully_read = false,
                MessageType::ATTRIBUTE => {
                    let Ok((attr, data_off)) =
                        AttributeMessage::parse_resolving_at_with_length_width(
                            &region[body..body_end],
                            widths.lengths,
                            &resolver,
                        )
                    else {
                        out.fully_read = false;
                        p = body_end;
                        continue;
                    };
                    out.holds_a_reference |=
                        datatype::datatype_holds_object_address(&attr.datatype);
                    if let Some(slots) = element_slots(&attr.datatype) {
                        edits.clear();
                        collect_slots(
                            &attr.datatype,
                            &slots,
                            &attr.raw_data,
                            body_at + data_off as u64,
                            base,
                            relocations,
                            &mut edits,
                        );
                        // An attribute's value lives inside the header, so it is
                        // edited as part of the chunk rather than on its own.
                        record_header_edits(plan, chunk.span, &edits);
                    }
                }
                _ => {}
            }
            p = body_end;
        }
    }

    let (Some(dt), Some((layout_body, layout_at))) = (element_dt, layout_msg) else {
        return Ok(out);
    };
    // Decide from the *datatype* whether this object's elements can hold an
    // address, before deciding to read them. Reading a data block first and
    // asking afterwards would charge each commit the whole file's raw data.
    let Some(slots) = element_slots(&dt) else {
        return Ok(out);
    };
    let Ok(dl) = DataLayout::parse_with_widths(layout_body, widths) else {
        return Ok(out);
    };
    match dl {
        DataLayout::Contiguous {
            address: Some(a),
            size,
        } => {
            // A contiguous data block sits outside the header and carries no
            // checksum, so its elements are patched with nothing else to fix.
            let Ok(at) = base.resolve(a).map(PhysicalAddress::get) else {
                return Ok(out);
            };
            let Ok(want) = usize::try_from(size) else {
                return Ok(out);
            };
            let Ok(raw) = src.read_exact_at(at, want) else {
                return Ok(out);
            };
            collect_slots(
                &dt,
                &slots,
                &raw,
                at,
                base,
                relocations,
                &mut plan.data_writes,
            );
        }
        DataLayout::Compact { data } => {
            edits.clear();
            collect_slots(
                &dt,
                &slots,
                &data,
                layout_at + COMPACT_DATA_OFFSET as u64,
                base,
                relocations,
                &mut edits,
            );
            // Compact elements are inline in the layout message, so they are
            // edited as part of the header chunk that message sits in.
            if let Some(span) = chunks.iter().map(|c| c.span).find(|range| {
                range.start() <= PhysicalAddress::new(layout_at)
                    && PhysicalAddress::new(layout_at) < range.end()
            }) {
                record_header_edits(plan, span, &edits);
            }
        }
        // Chunked and virtual layouts, and a contiguous dataset with no storage
        // allocated, hold nothing this reaches.
        _ => {}
    }
    Ok(out)
}

/// Records physical edit positions as byte offsets within their header chunk.
///
/// Skips edits below the chunk's start or whose displacement exceeds [`usize::MAX`].
/// [`Plan::apply`] checks each recorded header edit against its message region.
fn record_header_edits(plan: &mut Plan, span: FileRange, edits: &[(u64, StoredAddress)]) {
    for &(at, value) in edits {
        let Some(offset) = PhysicalAddress::new(at)
            .checked_distance_from(span.start())
            .and_then(|o| usize::try_from(o).ok())
        else {
            continue;
        };
        plan.header_writes
            .entry(span)
            .or_default()
            .push((offset, value));
    }
}

/// Where the 8-byte object references sit within one element of `dt`, or `None`
/// when there are none to rewrite.
///
/// [`datatype::embedded_reference_slots`] returns three outcomes, and this combines two of
/// them. `None` means the type reaches an address this cannot locate — wider
/// than eight bytes, a dataset-region reference, a variable length of them, an
/// enumeration over one. `Some([])` means the type holds no address at all. The
/// difference matters where a caller must *refuse* what it could not screen —
/// which is what `edit::screen_resolved_references` does, and why conflating the
/// two was a hole in the #317 screen — but nothing here refuses, so both answers
/// lead to the same silence.
///
/// Folding them anyway is what keeps a data block from being read for a dataset
/// with no slots to patch, which the walk would otherwise do once per dataset
/// per commit.
fn element_slots(dt: &Datatype) -> Option<Vec<usize>> {
    datatype::embedded_reference_slots(dt).filter(|s| !s.is_empty())
}

/// Whether an attribute of this datatype holds object references *this walk can
/// find and repoint*.
///
/// The same question [`element_slots`] answers for the scan itself, exposed so an
/// edit can refuse to move such an attribute somewhere the walk does not reach —
/// a dense attribute heap, which [`scan_object`] reads as unproven and collects
/// nothing from (issue #102).
///
/// Asking the walk's own predicate is what keeps that refusal from being wider
/// than the guarantee it protects. A reference this walk never repointed even
/// with the attribute in the object header — one wider than 8 bytes, a
/// dataset-region reference, a variable length of references — is no worse off in
/// a heap, and refusing to move it would buy nothing.
pub(crate) fn attribute_references_are_repointable(dt: &Datatype) -> bool {
    element_slots(dt).is_some()
}

/// Record a write for every 8-byte object reference in `raw` that names a
/// relocated header.
///
/// `raw_at` is the absolute file offset `raw` was read from, and `slots` is
/// [`element_slots`] for `dt`. Addresses are stored relative to the superblock
/// base, so each is shifted by `base` before being looked up and the replacement
/// is shifted back. Each edit is appended to `out` as
/// `(absolute file offset, value)`; where those bytes live is the caller's to
/// know, and decides whether they are written directly or resealed inside a
/// header chunk.
///
/// Finding the addresses is [`datatype::stored_object_references`], shared with
/// `edit::screen_resolved_references`: the two do different things with an
/// address and must not come to differ about where one is.
fn collect_slots(
    dt: &Datatype,
    slots: &[usize],
    raw: &[u8],
    raw_at: u64,
    base: AddressContext,
    relocations: &BTreeMap<u64, u64>,
    out: &mut Vec<(u64, StoredAddress)>,
) {
    for (offset, stored) in datatype::stored_object_references(raw, dt.type_size() as usize, slots)
    {
        // The two sentinels name no object and are never relocated; skipping
        // them keeps a zero-based file from matching a relocation whose old
        // address happened to be the base.
        if stored == 0 || stored == u64::MAX {
            continue;
        }
        let Ok(abs) = base
            .resolve(StoredAddress::new(stored))
            .map(PhysicalAddress::get)
        else {
            continue;
        };
        let Some(&new) = relocations.get(&abs) else {
            continue;
        };
        let Ok(value) = base.relative(PhysicalAddress::new(new)) else {
            continue;
        };
        out.push((raw_at + offset as u64, value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute::AttributeMessage;
    use crate::datatype::ReferenceType;
    use crate::edit;
    use crate::reader::OpenedSource;
    use crate::source::BytesSource;
    use crate::type_builders::{self, AttrValue};
    use test_util::attribute;
    use test_util::object_header::v2 as v2_bytes;
    use test_util::object_header::{Message, MessageType as RecordType};
    use test_util::widths::Widths;

    /// Where a hand-built header is placed in the test image. Non-zero so an
    /// offset accidentally taken relative to the *chunk* rather than the file —
    /// or the reverse — cannot pass.
    const HEADER_AT: u64 = 4096;

    /// Wrap a message body in the object-header record a header region holds it
    /// in: type, body size, flags, body.
    fn message_record(msg_type: MessageType, body: &[u8]) -> Vec<u8> {
        v2_bytes::message_record(
            &Message::new(RecordType(msg_type.to_u16()), body),
            v2_bytes::HeaderFlags::default(),
        )
    }

    /// A region of plain (4-byte-record) messages, the layout every writer in
    /// this crate emits.
    fn plain_region(bytes: Vec<u8>) -> crate::edit::OhRegion {
        crate::edit::OhRegion::new(
            bytes,
            hdf5_pure_format::__private::ObjectHeaderPrefix::PLAIN,
        )
    }

    /// An object-reference attribute named `name` pointing at `address`.
    ///
    /// Relabels a `u64` attribute, whose value is already the eight little-endian
    /// bytes an object reference is stored as. No public API stages one:
    /// [`AttrValue`] has no reference variant, so a file carrying such an
    /// attribute was written by libhdf5.
    fn reference_attr(name: &str, address: u64) -> AttributeMessage {
        let mut attr = type_builders::build_attr_message(name, &AttrValue::U64(address)).unwrap();
        attr.datatype = type_builders::make_object_reference_type();
        attr
    }

    /// A file image holding `region` as a version 2 object header at
    /// [`HEADER_AT`], with the space before it filled with a byte that is not
    /// part of any address.
    fn image_with_header(region: &[u8]) -> (BytesSource<Vec<u8>>, Vec<u8>) {
        let header =
            edit::object_header::build_v2_object_header(&plain_region(region.to_vec())).unwrap();
        let mut bytes = vec![0xAAu8; HEADER_AT as usize];
        bytes.extend_from_slice(&header);
        (BytesSource::new(bytes.clone()), bytes)
    }

    /// Apply a plan to an in-memory image, so a test can read back what it wrote.
    struct Bytes(Vec<u8>);
    impl PatchTarget for Bytes {
        fn read(&self, at: u64, len: usize) -> Result<Vec<u8>, Error> {
            let at = at as usize;
            Ok(self.0[at..at + len].to_vec())
        }
        fn write(&mut self, at: u64, bytes: &[u8]) -> Result<(), Error> {
            let at = at as usize;
            self.0[at..at + bytes.len()].copy_from_slice(bytes);
            Ok(())
        }
    }

    /// An Attribute Info (0x0015) message body: version, flags, then the fractal
    /// heap and name-index addresses, `None` encoding as the undefined address.
    /// Mirrors `file_writer::serialize_attribute_info`, whose compact form
    /// (`None`) is what this crate and the reference library attach to an object
    /// with ordinary inline attributes.
    fn attribute_info(fractal_heap: Option<u64>) -> Vec<u8> {
        attribute::info(fractal_heap, None, Widths::EIGHT)
    }

    /// A [`PatchTarget`] that records where each write started and how long it
    /// was, for the tests that are about the write *shape* rather than the bytes.
    struct Recording {
        bytes: Bytes,
        writes: Vec<(u64, usize)>,
    }
    impl PatchTarget for Recording {
        fn read(&self, at: u64, len: usize) -> Result<Vec<u8>, Error> {
            self.bytes.read(at, len)
        }
        fn write(&mut self, at: u64, bytes: &[u8]) -> Result<(), Error> {
            self.writes.push((at, bytes.len()));
            self.bytes.write(at, bytes)
        }
    }

    fn scan(src: &BytesSource<Vec<u8>>, relocations: &[(u64, u64)]) -> (Plan, Scanned) {
        let mut plan = Plan::default();
        let map: BTreeMap<u64, u64> = relocations.iter().copied().collect();
        let scanned = scan_object(
            &OpenedSource::for_test(src, AddressContext::from_effective_base(BaseAddress::ZERO)),
            HEADER_AT,
            crate::file_writer::WIDTHS,
            &map,
            &mut plan,
        )
        .unwrap();
        (plan, scanned)
    }

    #[test]
    fn an_inline_reference_attribute_is_repointed_and_the_chunk_resealed() {
        let region = message_record(
            MessageType::ATTRIBUTE,
            &reference_attr("target", 300)
                .serialize_v3(
                    crate::file_writer::OFFSET_WIDTH,
                    crate::file_writer::LENGTH_SIZE,
                )
                .unwrap(),
        );
        let (src, bytes) = image_with_header(&region);
        let (plan, scanned) = scan(&src, &[(300, 900)]);

        assert!(scanned.holds_a_reference);
        assert!(scanned.fully_read);
        assert_eq!(plan.len(), 1);
        // The edit is filed against the header chunk, not written on its own:
        // the checksum covering it has to move with it.
        assert!(plan.data_writes.is_empty());
        assert_eq!(plan.header_writes.len(), 1);
        // The address arithmetic is checked against the file rather than
        // restated: the eight bytes the plan is about to overwrite must be the
        // ones that hold the old address today. An offset taken from the wrong
        // origin — the message body, the chunk, the region — lands somewhere
        // that does not.
        let site = plan.sites()[0] as usize;
        assert_eq!(
            u64::from_le_bytes(bytes[site..site + 8].try_into().unwrap()),
            300,
            "the plan must aim at the bytes that hold the address it read"
        );

        let mut target = Bytes(bytes.clone());
        plan.apply(&mut target).unwrap();
        // The whole chunk was rewritten, so the file is longer than the header
        // only where it was before.
        assert_eq!(target.0.len(), bytes.len());
        // The stored address now names the new header, and the chunk's checksum
        // agrees with its own bytes: re-scanning finds nothing left to do.
        let rescanned = BytesSource::new(target.0.clone());
        let (again, _) = scan(&rescanned, &[(300, 900)]);
        assert!(again.is_empty(), "the address should already be 900");
        let (moved_again, _) = scan(&rescanned, &[(900, 1500)]);
        assert_eq!(moved_again.len(), 1, "and it should now read as 900");
        assert_checksum_holds(&target.0);
    }

    /// The trailing four bytes of the header chunk are the Jenkins checksum of
    /// everything before them.
    fn assert_checksum_holds(bytes: &[u8]) {
        let chunk = &bytes[HEADER_AT as usize..];
        let body = &chunk[..chunk.len() - 4];
        assert_eq!(
            u32::from_le_bytes(chunk[chunk.len() - 4..].try_into().unwrap()),
            checksum::jenkins_lookup3(body),
            "the header chunk must be resealed after its bytes change"
        );
    }

    #[test]
    fn a_compact_reference_dataset_is_repointed_in_place() {
        // A compact dataset keeps its elements inside the data-layout message
        // rather than in a data block. No writer in this crate emits one, so
        // this is the only way the path is reached at all.
        let mut region = message_record(
            MessageType::DATATYPE,
            &hdf5_pure_format::__private::serialize_datatype(
                &type_builders::make_object_reference_type(),
            )
            .unwrap(),
        );
        let mut layout = vec![3u8, 0];
        layout.extend_from_slice(&8u16.to_le_bytes());
        layout.extend_from_slice(&300u64.to_le_bytes());
        region.extend_from_slice(&message_record(MessageType::DATA_LAYOUT, &layout));

        let (src, bytes) = image_with_header(&region);
        let (plan, scanned) = scan(&src, &[(300, 900)]);
        assert!(scanned.holds_a_reference);
        assert_eq!(plan.len(), 1);
        assert!(plan.data_writes.is_empty(), "compact data is in the header");

        let mut target = Bytes(bytes);
        plan.apply(&mut target).unwrap();
        assert_checksum_holds(&target.0);
        let (again, _) = scan(&BytesSource::new(target.0.clone()), &[(900, 1500)]);
        assert_eq!(again.len(), 1);
    }

    #[test]
    fn the_two_undefined_addresses_are_never_repointed() {
        // A slot holding zero or the all-ones sentinel names no object. Zero in
        // particular would otherwise match a relocation whose vacated address
        // happened to be the superblock base.
        for sentinel in [0u64, u64::MAX] {
            let region = message_record(
                MessageType::ATTRIBUTE,
                &reference_attr("target", sentinel)
                    .serialize_v3(
                        crate::file_writer::OFFSET_WIDTH,
                        crate::file_writer::LENGTH_SIZE,
                    )
                    .unwrap(),
            );
            let (src, _) = image_with_header(&region);
            let (plan, scanned) = scan(&src, &[(sentinel, 900), (0, 900)]);
            assert!(
                plan.is_empty(),
                "the sentinel {sentinel:#x} names no object and must be left alone"
            );
            assert!(
                scanned.holds_a_reference,
                "it is still a reference datatype"
            );
        }
    }

    #[test]
    fn a_reference_this_cannot_address_leaves_the_file_unproven_and_untouched() {
        // A 16-byte reference is one `embedded_reference_slots` declines to map.
        // The object still reports that it holds a reference, so no commit is
        // ever licensed to skip the walk on this file — and nothing is written,
        // because the address could not be located to begin with.
        let mut attr = type_builders::build_attr_message("target", &AttrValue::U64(300)).unwrap();
        attr.datatype = Datatype::Reference {
            size: 16,
            ref_type: ReferenceType::Object,
        };
        attr.raw_data = vec![0u8; 16];
        let region = message_record(
            MessageType::ATTRIBUTE,
            &attr
                .serialize_v3(
                    crate::file_writer::OFFSET_WIDTH,
                    crate::file_writer::LENGTH_SIZE,
                )
                .unwrap(),
        );
        let (src, _) = image_with_header(&region);
        let (plan, scanned) = scan(&src, &[(300, 900)]);
        assert!(plan.is_empty());
        assert!(
            scanned.holds_a_reference,
            "an unmappable reference is still a reference: the file must never be \
             proved free of them"
        );
    }

    #[test]
    fn dense_attribute_storage_leaves_the_object_unproven() {
        // An object storing its attributes in a fractal heap presents no
        // Attribute message in its header, so nothing here can see what they
        // hold. Reading that as "no references in this object" is what would let
        // a later commit skip the walk over a file full of them.
        let dense = message_record(MessageType::ATTRIBUTE_INFO, &attribute_info(Some(4096)));
        let (src, _) = image_with_header(&dense);
        let (plan, scanned) = scan(&src, &[(300, 900)]);
        assert!(plan.is_empty(), "there is nothing here this can address");
        assert!(
            !scanned.fully_read,
            "a dense attribute set is unread, so the object cannot be counted \
             towards a file proved free of references"
        );

        // The same message with an *undefined* heap address is the compact form
        // nearly every object the C library writes carries, and it hides nothing.
        let compact = message_record(MessageType::ATTRIBUTE_INFO, &attribute_info(None));
        let (src, _) = image_with_header(&compact);
        let (_, scanned) = scan(&src, &[(300, 900)]);
        assert!(
            scanned.fully_read,
            "an Attribute Info message is not dense storage by itself"
        );
    }

    #[test]
    fn a_shared_attribute_record_leaves_the_object_unproven() {
        // An attribute held in the file's shared-message table is not in this
        // header, so nothing here can see what it holds. Reading that as "no
        // references in this object" is what would let a later commit skip the
        // walk over a file that needs it.
        let attr = reference_attr("target", 300)
            .serialize_v3(
                crate::file_writer::OFFSET_WIDTH,
                crate::file_writer::LENGTH_SIZE,
            )
            .unwrap();
        let mut record = message_record(MessageType::ATTRIBUTE, &attr);
        record[3] = MessageFlags::SHARED.get();
        let (src, _) = image_with_header(&record);
        let (plan, scanned) = scan(&src, &[(300, 900)]);
        assert!(
            plan.is_empty(),
            "the message body is a pointer into the shared table, not the value"
        );
        assert!(
            !scanned.fully_read,
            "an unread attribute leaves the object unproven"
        );
    }

    #[test]
    fn a_header_chunk_is_written_from_the_byte_that_changed() {
        // The read covers the whole chunk, because the checksum does; the write
        // must not, or repointing one attribute near the end of a wide header
        // rewrites the header. Borrowed wholesale from `publish_checksummed`,
        // and worth its own assertion for the same reason that one has: the
        // write *count* is identical either way, so nothing else notices.
        let region = message_record(
            MessageType::ATTRIBUTE,
            &reference_attr("target", 300)
                .serialize_v3(
                    crate::file_writer::OFFSET_WIDTH,
                    crate::file_writer::LENGTH_SIZE,
                )
                .unwrap(),
        );
        let (src, bytes) = image_with_header(&region);
        let (plan, _) = scan(&src, &[(300, 900)]);
        let site = plan.sites()[0];

        let mut target = Recording {
            bytes: Bytes(bytes),
            writes: Vec::new(),
        };
        plan.apply(&mut target).unwrap();
        assert_eq!(
            target.writes.len(),
            1,
            "one chunk, one write — value and checksum together"
        );
        assert_eq!(
            target.writes[0].0, site,
            "the write starts at the repointed address, not at the chunk"
        );
    }

    #[test]
    fn a_committed_datatype_is_resolved_through_the_base_address() {
        for base in [
            AddressContext::from_effective_base(BaseAddress::ZERO),
            AddressContext::from_effective_base(BaseAddress::new(1024)),
        ] {
            const TYPE_AT: u64 = 2048;
            let committed =
                edit::object_header::build_v2_object_header(&plain_region(message_record(
                    MessageType::DATATYPE,
                    &hdf5_pure_format::__private::serialize_datatype(
                        &type_builders::make_object_reference_type(),
                    )
                    .unwrap(),
                )))
                .unwrap();

            // The dataset's own header: a *shared* datatype message naming the
            // committed object, and a compact element holding the address.
            let mut shared = message_record(
                MessageType::DATATYPE,
                &crate::shared_message::encode_committed_ref(
                    base.relative(PhysicalAddress::new(TYPE_AT)).unwrap(),
                    crate::file_writer::OFFSET_WIDTH,
                ),
            );
            shared[3] = MessageFlags::SHARED.get();
            let mut layout = vec![3u8, 0];
            layout.extend_from_slice(&8u16.to_le_bytes());
            layout.extend_from_slice(&(300u64).to_le_bytes());
            shared.extend_from_slice(&message_record(MessageType::DATA_LAYOUT, &layout));
            let dataset =
                edit::object_header::build_v2_object_header(&plain_region(shared)).unwrap();

            let mut bytes = vec![0xAAu8; TYPE_AT as usize];
            bytes.extend_from_slice(&committed);
            bytes.resize(HEADER_AT as usize, 0xAA);
            bytes.extend_from_slice(&dataset);

            let mut plan = Plan::default();
            let map = BTreeMap::from([(
                300 + base.effective_base().get(),
                900 + base.effective_base().get(),
            )]);
            let scanned = scan_object(
                &OpenedSource::for_test(&BytesSource::new(bytes), base),
                HEADER_AT,
                crate::file_writer::WIDTHS,
                &map,
                &mut plan,
            )
            .unwrap();
            assert!(
                scanned.holds_a_reference,
                "base {base:?}: the committed type must be followed far enough to \
                 see it names a reference"
            );
            assert_eq!(
                plan.len(),
                1,
                "base {base:?}: the element must be repointed"
            );
        }
    }

    /// A header the commit vacated is never scanned, even when a link still
    /// names it.
    ///
    /// Walking the committed tree is meant to make that unreachable, and it does
    /// — unless a link the commit *failed* to repoint still points at the old
    /// header. A group with two hard links is rebuilt today with only its parent
    /// link patched, so the alias is exactly such a link. The vacated span is in
    /// the free list by the time this runs, so an edit filed against it is a
    /// write into space the next allocation may take.
    #[test]
    fn a_vacated_header_is_not_scanned_even_when_a_link_still_names_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vacated.h5");
        let mut b = crate::writer::FileBuilder::new();
        b.create_dataset("d").with_i32_data(&[1, 2, 3]);
        b.create_dataset("refs").with_path_references(&["d"]);
        b.write(&path).unwrap();

        let file = crate::File::open(&path).unwrap();
        let superblock = file.superblock().clone();
        let root = AddressContext::from_effective_base(superblock.base_address)
            .resolve(superblock.root_group_address)
            .unwrap()
            .get();
        // What `refs` stores, which is `d`'s header address. The file has no
        // userblock, so the stored base-relative address is the absolute one.
        let stored = u64::from_le_bytes(
            file.dataset("refs").unwrap().read_raw().unwrap()[..8]
                .try_into()
                .unwrap(),
        );
        drop(file);
        let source = crate::source::BytesSource::new(std::fs::read(&path).unwrap());

        // The control: `d` has moved, the root has not, so the walk descends
        // through the root and repoints the one element in `refs`.
        let moved = BTreeMap::from([(stored, stored + 4096)]);
        let opened = OpenedSource::for_test(
            &source,
            AddressContext::from_effective_base(superblock.base_address),
        );
        let reached = super::plan(
            &opened,
            &superblock,
            FormatWidths {
                offsets: superblock.offset_size,
                lengths: superblock.length_size,
            },
            &moved,
            1 << 20,
        )
        .unwrap();
        assert_eq!(reached.len(), 1, "the walk must reach `refs` at all");

        // The same file and the same target, with the root itself vacated. The
        // root is the only way in, so the guard makes the whole file
        // unreachable — and the edit the control found must not be made, because
        // reaching it meant walking a header that is no longer part of any tree.
        let mut vacated = moved.clone();
        vacated.insert(root, root + 8192);
        let skipped = super::plan(
            &opened,
            &superblock,
            FormatWidths {
                offsets: superblock.offset_size,
                lengths: superblock.length_size,
            },
            &vacated,
            1 << 20,
        )
        .unwrap();
        assert!(
            skipped.is_empty(),
            "a vacated header must not be scanned, nor descended through"
        );
        assert!(
            !skipped.proved_free_of_references(),
            "and skipping it leaves the file unproven rather than proven clean"
        );
    }

    #[test]
    fn an_address_no_relocation_names_is_left_alone() {
        let region = message_record(
            MessageType::ATTRIBUTE,
            &reference_attr("target", 300)
                .serialize_v3(
                    crate::file_writer::OFFSET_WIDTH,
                    crate::file_writer::LENGTH_SIZE,
                )
                .unwrap(),
        );
        let (src, _) = image_with_header(&region);
        let (plan, _) = scan(&src, &[(301, 900), (299, 900)]);
        assert!(
            plan.is_empty(),
            "only the exact vacated address is repointed; a neighbour is a \
             different object"
        );
    }
}
