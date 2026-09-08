//! The two guards that decide who may open a file: OS advisory locking for the
//! in-place editor (issue #73), and the superblock's durable status-flags byte
//! (issue #245).
//!
//! Locking is the crash-safe half of HDF5's concurrency model and the `hdf5-pure`
//! analogue of `H5Pset_file_locking` / the `HDF5_USE_FILE_LOCKING` environment
//! variable. It is deliberately distinct from the *superblock consistency flag*
//! (the durable `status_flags` byte a SWMR writer sets; see [`crate::File::open_swmr_writer`]):
//!
//! - An **OS lock** is owned by the kernel and tied to the open file. It is
//!   released automatically when the process exits *for any reason* — clean exit,
//!   panic, `SIGKILL`, even power loss — so it never leaves stale state and is
//!   the authoritative signal for "a writer is alive *right now*".
//! - The **on-disk flag** is just a byte; only userspace code at clean shutdown
//!   can reset it, so a crash freezes it set. Recover it with
//!   [`crate::File::clear_swmr_flag`] (the `h5clear -s` equivalent).
//!   A crash freezing it set is also what makes it useful beyond SWMR — see
//!   [`WRITE_ACCESS`], which a page-buffered session raises for its lifetime.
//!
//! Both are enforced here: [`acquire_exclusive`] takes the lock, and
//! [`check_status_flags`] refuses an open the on-disk byte says is unsafe
//! (issue #245). The two cover different windows — the lock catches a live
//! writer in this or another process, the flag catches one that is live *or*
//! crashed, including a SWMR writer that holds no lock at all.
//!
//! ## Lock scope: the in-place editor only
//!
//! Only [`crate::File::open_rw`] (and the [`crate::File::clear_swmr_flag`]
//! recovery rewrite) take a lock — an **exclusive** one — so a second editor or
//! a concurrent writer cannot open the file. [`crate::File::open_swmr_writer`] and the
//! readers ([`crate::File::open`] and friends) take **no** lock, on purpose:
//!
//! - SWMR is single-writer-*by-contract* and is designed for concurrent reads;
//!   the reference library itself runs SWMR with file locking disabled. Holding
//!   a lock would defeat the "multiple-reader" half.
//! - Crucially, [`std::fs::File`] locking is **advisory on Unix** (`flock`) but
//!   **mandatory on Windows** (`LockFileEx`): a held lock there blocks *reads* by
//!   every other handle, not just other lock attempts. A whole-file lock on a
//!   SWMR writer would therefore make the file unreadable to its readers on
//!   Windows. Confining locking to the exclusive editor keeps reads working on
//!   every platform. (One consequence: while an editor holds the lock, a
//!   concurrent read of the same file is permitted on Unix but blocked by the OS
//!   on Windows — drop the editor before reading the file back.)
//!
//! Locking uses the cross-platform [`std::fs::File`] lock API, so it adds no
//! dependency, and it lives only in the already `std`-gated edit path, so
//! `no_std`/`wasm` builds are unaffected.

use std::fs::{File, TryLockError};
use std::path::Path;

use crate::error::Error;
use crate::superblock::Superblock;

/// Policy for OS advisory file locking when opening a file for editing.
///
/// The default is [`FileLocking::Enabled`]. The `HDF5_USE_FILE_LOCKING`
/// environment variable, when set to a recognized value, overrides the requested
/// policy (matching the reference HDF5 library): `FALSE`/`0`/`NO`/`OFF` disable
/// locking, `BEST_EFFORT` selects best-effort, and `TRUE`/`1`/`YES`/`ON` enable
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FileLocking {
    /// Acquire the lock, and fail the open with [`Error::Io`] if the filesystem
    /// does not support locking. A lock held by another process always fails the
    /// open with [`Error::FileLocked`].
    #[default]
    Enabled,
    /// Do not attempt to lock the file at all.
    Disabled,
    /// Attempt to lock, but proceed *without* a lock when the filesystem reports
    /// that locking is unavailable (e.g. some NFS / network mounts). A lock that
    /// is genuinely *held* by another process still fails the open. Mirrors the
    /// reference library's `BEST_EFFORT` / `ignore_disabled_locks`.
    BestEffort,
}

/// Parse a recognized `HDF5_USE_FILE_LOCKING` value into a policy, or `None` for
/// an unrecognized value (in which case the requested policy is kept).
///
/// Pure (no environment access) so it can be unit-tested without the
/// process-global, edition-2024-`unsafe` env mutators.
fn parse_env(value: &str) -> Option<FileLocking> {
    let v = value.trim();
    if v.eq_ignore_ascii_case("FALSE")
        || v == "0"
        || v.eq_ignore_ascii_case("NO")
        || v.eq_ignore_ascii_case("OFF")
    {
        Some(FileLocking::Disabled)
    } else if v.eq_ignore_ascii_case("BEST_EFFORT") {
        Some(FileLocking::BestEffort)
    } else if v.eq_ignore_ascii_case("TRUE")
        || v == "1"
        || v.eq_ignore_ascii_case("YES")
        || v.eq_ignore_ascii_case("ON")
    {
        Some(FileLocking::Enabled)
    } else {
        None
    }
}

/// Apply the `HDF5_USE_FILE_LOCKING` environment override to a requested policy.
/// The environment variable, when set to a recognized value, takes precedence.
fn resolve(requested: FileLocking) -> FileLocking {
    std::env::var("HDF5_USE_FILE_LOCKING")
        .ok()
        .and_then(|v| parse_env(&v))
        .unwrap_or(requested)
}

/// Acquire an **exclusive** advisory lock on `handle` for a writer open.
///
/// Non-blocking: if another process holds a conflicting lock, this returns
/// [`Error::FileLocked`] immediately rather than waiting. The lock is released
/// when `handle` is dropped (or the process exits, including on a crash).
pub(crate) fn acquire_exclusive(
    handle: &File,
    requested: FileLocking,
    path: &Path,
) -> Result<(), Error> {
    let mode = resolve(requested);
    if mode == FileLocking::Disabled {
        return Ok(());
    }
    match handle.try_lock() {
        Ok(()) => Ok(()),
        // A conflicting lock is genuinely held by another process: the file is
        // in use. `BestEffort` does not soften this — only *unavailable* locking
        // is tolerated, not active contention.
        Err(TryLockError::WouldBlock) => Err(Error::FileLocked(format!(
            "{}: file is already locked by another process. If a previous writer \
             crashed, the OS lock is released automatically (try again); a leftover \
             on-disk SWMR flag can be cleared with File::clear_swmr_flag. Set \
             HDF5_USE_FILE_LOCKING=FALSE or pass FileLocking::Disabled to bypass locking.",
            path.display(),
        ))),
        // Locking failed for another reason — typically the filesystem does not
        // support advisory locks (some NFS / network mounts).
        Err(TryLockError::Error(e)) => match mode {
            FileLocking::BestEffort => Ok(()),
            _ => Err(Error::Io(e)),
        },
    }
}

/// Superblock status-flag bit 0 (`H5F_SUPER_WRITE_ACCESS`): the file is open
/// for write access. The reference C library raises it for *any* writer; this
/// crate raises it for two:
///
/// - [`crate::File::open_swmr_writer`], alongside [`SWMR_WRITE_ACCESS`];
/// - a session given a page buffer
///   ([`crate::FileAccessProperties::with_page_buffer_size`]), which raises this
///   bit alone. That session holds dirty pages across the write engine's
///   ordering barriers, so a process that died mid-flush could leave a file that
///   reads clean and returns the wrong bytes; the mark makes it a file every
///   reader refuses instead (issue #308), until one says with
///   [`WriteMarkPolicy::AllowSnapshot`] that it knows what it is reading.
///
/// An ordinary [`crate::File::open_rw`] session raises nothing, and is guarded by
/// the OS lock alone.
pub(crate) const WRITE_ACCESS: u32 = 0x01;

/// Superblock status-flag bit 2 (`H5F_SUPER_SWMR_WRITE_ACCESS`): the writer
/// holding the file is a SWMR writer, so a SWMR reader may attach to it.
pub(crate) const SWMR_WRITE_ACCESS: u32 = 0x04;

/// Policy for a read-only open of a file whose superblock marks it as open for
/// write by a writer that is *not* a SWMR writer — superblock status-flag bit 0
/// (`H5F_SUPER_WRITE_ACCESS`) alone, the mark a page-buffered session
/// ([`crate::FileAccessProperties::with_page_buffer_size`]) holds for its life.
///
/// Set it with
/// [`FileAccessProperties::with_write_mark_policy`](crate::FileAccessProperties::with_write_mark_policy),
/// which states when the assertion [`AllowSnapshot`](Self::AllowSnapshot) makes
/// is true. It governs the read-only opens only: a SWMR pair is followed with
/// [`crate::File::open_swmr`] whatever this says, and no value of it lets a
/// second writer join a file a writer holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WriteMarkPolicy {
    /// Refuse the open with [`Error::FileMarkedInUse`]. The default, and what
    /// `H5Fopen` does with the same byte.
    #[default]
    Refuse,
    /// Read the file as it stands, on the caller's assertion that the writer
    /// has flushed: it called [`crate::File::sync`], or it stopped after a flush
    /// and the mark stands only because nothing cleared it. Every other refusal
    /// stays in place.
    AllowSnapshot,
}

/// What an open intends to do with the file, selecting which status-flag
/// combinations [`check_status_flags`] refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenIntent {
    /// A plain read — [`crate::File::open`] and
    /// [`crate::File::open_streaming`], the C library's `H5F_ACC_RDONLY`.
    ///
    /// Carries the caller's [`WriteMarkPolicy`], which is the one intent it can
    /// mean anything for: the other two have no snapshot to take.
    Read(WriteMarkPolicy),
    /// A SWMR read — [`crate::File::open_swmr`], the C library's
    /// `H5F_ACC_RDONLY | H5F_ACC_SWMR_READ`.
    SwmrRead,
    /// An open that may modify the file — [`crate::File::open_rw`] and
    /// [`crate::File::open_swmr_writer`], the C library's `H5F_ACC_RDWR` with or
    /// without `H5F_ACC_SWMR_WRITE`.
    Write,
}

/// What an open was asked to open, for the refusal to name.
///
/// The companion of [`OpenIntent`]: that says why the file was being opened,
/// this says what the file *was*. Neither reaches the decision — the superblock
/// makes that alone — but both shape what the error can usefully say, and the
/// difference is not only cosmetic. Two of the three recoveries a refused
/// snapshot read would otherwise be told to try, [`crate::File::open_swmr`] and
/// [`crate::File::clear_swmr_flag`], need a filesystem path; naming them to a
/// caller who opened a byte source would be advice it cannot take.
///
/// A closed set rather than a `&dyn Display`, so the wording every refusal
/// leads with lives here beside the reasons it is concatenated with, and a
/// third kind of open arrives as a variant a reviewer sees.
#[derive(Debug, Clone, Copy)]
pub(crate) enum OpenTarget<'a> {
    /// A file opened by name, which the refusal reports and whose flag
    /// [`crate::File::clear_swmr_flag`] can clear in place.
    Path(&'a Path),
    /// Bytes handed to [`crate::File::from_source`], which have no name and no
    /// in-place recovery.
    Source,
}

impl OpenTarget<'_> {
    /// What a caller refused a snapshot read can do instead, which depends on
    /// both the target and the mark.
    ///
    /// The target, because a source has no path for `open_swmr` to follow or for
    /// `clear_swmr_flag` to write to. The mark, because `open_swmr` follows a
    /// SWMR writer and nothing else: naming it to a caller refused by
    /// [`WRITE_ACCESS`] alone would name a recovery that fails in its turn, the
    /// mismatched pair a SWMR reader refuses (issue #419). That caller's opt-in
    /// is [`WriteMarkPolicy::AllowSnapshot`], which is what this names instead.
    ///
    /// `swmr_claimed` is the SWMR bit, not the pair: a file carrying it without
    /// [`WRITE_ACCESS`] claims a SWMR writer this crate's opt-in deliberately
    /// does not unlock, so it is pointed at the SWMR reader — whose own refusal
    /// then names the inconsistency — rather than at an opt-in that would refuse
    /// it again.
    fn read_recovery(self, swmr_claimed: bool) -> &'static str {
        match (self, swmr_claimed) {
            (Self::Path(_), true) => {
                "Use File::open_swmr to follow a live SWMR writer, or File::from_bytes to read \
                 the bytes as they stand; if a writer exited without closing the file, clear \
                 the flag with File::clear_swmr_flag"
            }
            (Self::Source, true) => {
                "Use File::from_bytes to read the bytes as they stand; following a live writer \
                 with File::open_swmr, and clearing a flag a writer left behind with \
                 File::clear_swmr_flag, both need a filesystem path"
            }
            (Self::Path(_), false) => {
                "No SWMR writer holds it, so no reader can follow it as one. Pass \
                 FileAccessProperties::with_write_mark_policy(WriteMarkPolicy::AllowSnapshot) to \
                 read it as it stands, which is a consistent snapshot once the writer has called \
                 File::sync or closed the file, or use File::from_bytes to read the bytes as they \
                 stand; if a writer exited without closing the file, clear the flag with \
                 File::clear_swmr_flag"
            }
            (Self::Source, false) => {
                "No SWMR writer holds it, so no reader can follow it as one. Pass \
                 FileAccessProperties::with_write_mark_policy(WriteMarkPolicy::AllowSnapshot) to \
                 read it as it stands, which is a consistent snapshot once the writer has called \
                 File::sync or closed the file, or use File::from_bytes to read the bytes as they \
                 stand; clearing a flag a writer left behind with File::clear_swmr_flag needs a \
                 filesystem path"
            }
        }
    }
}

impl core::fmt::Display for OpenTarget<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Path(path) => write!(f, "{}", path.display()),
            Self::Source => f.write_str("a byte source"),
        }
    }
}

/// Refuse an open the superblock's status-flags byte says is unsafe, matching
/// `H5F_open`'s check of the same byte (issue #245).
///
/// The byte records that a writer holds the file. It is *durable*, so it means
/// either "a writer is active right now" or "a writer exited without clearing
/// it" — the two are indistinguishable from the byte alone, which is why the
/// refusal names [`crate::File::clear_swmr_flag`] as the recovery rather than
/// guessing. The rules, one per intent:
///
/// - [`Write`](OpenIntent::Write) refuses either bit: a second writer must not
///   join a file a writer already holds. This is the one case an OS lock does
///   not already cover, because a SWMR writer takes no lock.
/// - [`Read`](OpenIntent::Read) refuses either bit: a plain reader buffers a
///   snapshot with no protocol for a writer mutating the file underneath it. To
///   follow a live SWMR writer, use [`crate::File::open_swmr`]. The one way past
///   it is [`WriteMarkPolicy::AllowSnapshot`], which the intent carries: it
///   admits a read of [`WRITE_ACCESS`] *alone* — the mark a page-buffered
///   session leaves, which no SWMR reader can follow (issue #419) — on the
///   caller's assertion that the writer has flushed. It does not admit a SWMR
///   pair, which has a reader of its own.
/// - [`SwmrRead`](OpenIntent::SwmrRead) refuses only a *mismatched* pair — one
///   bit without the other. Both bits is exactly the live SWMR writer it exists
///   to follow, and neither is a quiescent file.
///
/// ## Why only a version-3 superblock
///
/// The check is gated to superblock version 3 and up because that is where the
/// C library gates it, and a divergence in either direction is a real cost: the
/// C library raises the write bit on a version-0/1/2 file too and never reads
/// it back, so checking those versions would refuse files `H5Fopen` accepts —
/// every file left behind by a crashed C writer that predates SWMR. Nothing is
/// lost by matching it: both paths that raise a flag here require a version-3
/// superblock — SWMR writing because both libraries do (see
/// [`crate::File::open_swmr_writer`]), and a page buffer because it refuses to
/// buffer behind a mark no reader would honor — so no flag this crate raises
/// falls outside the gate.
///
/// `target` does not enter the decision, which the superblock and the intent's
/// own policy make between them. It shapes the error: what to call the file that
/// was refused, and which recoveries to name, since some of them need a
/// filesystem path.
pub(crate) fn check_status_flags(
    superblock: &Superblock,
    intent: OpenIntent,
    target: OpenTarget<'_>,
) -> Result<(), Error> {
    if superblock.version < 3 {
        return Ok(());
    }
    let flags = superblock.consistency_flags;
    let write = flags & WRITE_ACCESS != 0;
    let swmr = flags & SWMR_WRITE_ACCESS != 0;
    // Each arm states the whole condition: "marked open for write" describes two
    // of the three, and reporting the SWMR-read mismatch that way would misname
    // the case where only the SWMR bit is set.
    let reason = match intent {
        OpenIntent::Write if write || swmr => format!(
            "the superblock marks the file as open for write (status flags {flags:#04x}), so \
             another writer holds it. Open it read-only, or — if a writer exited without \
             closing the file — clear the flag with File::clear_swmr_flag"
        ),
        // The opt-in is checked inside the arm rather than in its guard so that
        // the arm keeps stating the whole rule in one place: what the flags say,
        // and the one assertion that overrides it.
        OpenIntent::Read(policy) if write || swmr => {
            if policy == WriteMarkPolicy::AllowSnapshot && !swmr {
                return Ok(());
            }
            format!(
                "the superblock marks the file as open for write (status flags {flags:#04x}), so \
                 a snapshot read is not safe. {}",
                target.read_recovery(swmr)
            )
        }
        OpenIntent::SwmrRead if write != swmr => format!(
            "the superblock's status flags disagree ({flags:#04x}): a SWMR reader needs a SWMR \
             writer (both the write and SWMR-write bits) or a quiescent file (neither). Clear \
             them with File::clear_swmr_flag if a writer exited without closing the file"
        ),
        _ => return Ok(()),
    };
    Err(Error::FileMarkedInUse(format!("{target}: {reason}.")))
}

/// Clear a stale status flag left in `path` by a writer that exited without a
/// clean close — the `h5clear -s` equivalent, behind
/// [`File::clear_swmr_flag`](crate::File::clear_swmr_flag). Safe to call on a
/// file whose flag is already clear.
///
/// It clears the byte whole, so it recovers a page-buffered session's crash mark
/// ([`WRITE_ACCESS`]) as well as a SWMR writer's pair. What it recovers is
/// *access*, not correctness: a page-buffered writer that crashed may have left
/// the file inconsistent in ways no checksum shows, which is the whole reason
/// the mark stands. `h5clear` makes the same trade.
///
/// This is the one recovery rewrite that takes the exclusive lock without going
/// through the editor, which is why it lives beside the locking policy it
/// depends on.
pub(crate) fn clear_swmr_flag_at(path: &Path) -> Result<(), Error> {
    use crate::signature;
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom, Write};

    let mut w = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(Error::Io)?;
    // Refuse to clear the flag out from under a live writer: an exclusive
    // lock here fails with `FileLocked` if another writer still holds the
    // file. A stale flag from a *crashed* writer has no live lock, so this
    // succeeds and the recovery proceeds.
    acquire_exclusive(&w, FileLocking::Enabled, path)?;
    let mut data = Vec::new();
    w.read_to_end(&mut data).map_err(Error::Io)?;
    let sig = signature::find_signature(&data)?;
    let mut sb = Superblock::parse(&data, sig)?;
    if sb.version < 2 {
        // `Superblock::serialize` emits the v2/v3 layout, so rewriting a
        // v0/v1 superblock here would corrupt it. This crate never SWMR-flags
        // a v0/v1 file, so there is nothing to clear; treat it as already
        // clean rather than risk a destructive rewrite.
        return Ok(());
    }
    if sb.consistency_flags == 0 {
        return Ok(());
    }
    sb.consistency_flags = 0;
    let bytes = sb.serialize();
    w.seek(SeekFrom::Start(sig as u64)).map_err(Error::Io)?;
    w.write_all(&bytes).map_err(Error::Io)?;
    w.sync_data().map_err(Error::Io)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::width::LengthWidth;
    use crate::width::OffsetWidth;

    #[test]
    fn parse_env_recognizes_disable_values() {
        for v in ["FALSE", "false", "0", "No", "off", " false "] {
            assert_eq!(parse_env(v), Some(FileLocking::Disabled), "value {v:?}");
        }
    }

    #[test]
    fn parse_env_recognizes_enable_and_best_effort() {
        for v in ["TRUE", "true", "1", "Yes", "on"] {
            assert_eq!(parse_env(v), Some(FileLocking::Enabled), "value {v:?}");
        }
        assert_eq!(parse_env("BEST_EFFORT"), Some(FileLocking::BestEffort));
        assert_eq!(parse_env("best_effort"), Some(FileLocking::BestEffort));
    }

    #[test]
    fn parse_env_unrecognized_is_none() {
        assert_eq!(parse_env(""), None);
        assert_eq!(parse_env("maybe"), None);
        assert_eq!(parse_env("2"), None);
    }

    #[test]
    fn default_is_enabled() {
        assert_eq!(FileLocking::default(), FileLocking::Enabled);
    }

    /// A minimal superblock carrying `version` and `flags`; every other field is
    /// irrelevant to the status-flag rules.
    fn flagged(version: u8, flags: u32) -> Superblock {
        Superblock {
            version,
            offset_size: OffsetWidth::Eight.get(),
            offset_width: OffsetWidth::Eight,
            length_size: LengthWidth::Eight.get(),
            length_width: LengthWidth::Eight,
            base_address: crate::address::BaseAddress::ZERO,
            eof_address: 0,
            root_group_address: 0,
            group_leaf_node_k: None,
            group_internal_node_k: None,
            indexed_storage_internal_node_k: None,
            free_space_address: None,
            driver_info_address: None,
            consistency_flags: flags,
            superblock_extension_address: None,
            checksum: None,
        }
    }

    /// A snapshot read under the default policy, which is what every rule
    /// below is stated against unless it names the opt-in.
    const READ: OpenIntent = OpenIntent::Read(WriteMarkPolicy::Refuse);

    fn allows(version: u8, flags: u32, intent: OpenIntent) -> bool {
        check_status_flags(
            &flagged(version, flags),
            intent,
            OpenTarget::Path(Path::new("f.h5")),
        )
        .is_ok()
    }

    /// The full rule, stated per intent rather than per flag value, so a table
    /// entry that moves has to move for a reason someone can name.
    #[test]
    fn status_flag_rules_per_intent() {
        for flags in [0x00, WRITE_ACCESS, SWMR_WRITE_ACCESS, 0x05] {
            let held = flags != 0;
            assert_eq!(
                allows(3, flags, OpenIntent::Write),
                !held,
                "a writer may open a file only when no flag claims it (flags {flags:#04x})"
            );
            assert_eq!(
                allows(3, flags, READ),
                !held,
                "a snapshot read is refused whenever a writer holds the file (flags {flags:#04x})"
            );
            assert_eq!(
                allows(3, flags, OpenIntent::SwmrRead),
                flags == 0x00 || flags == 0x05,
                "a SWMR reader needs both bits or neither (flags {flags:#04x})"
            );
        }
    }

    /// Bit 1 (`H5F_SUPER_FILE_OK`) is not one of the two the C library consults,
    /// so a file carrying only it opens for any intent.
    #[test]
    fn the_file_ok_bit_alone_refuses_nothing() {
        for intent in [READ, OpenIntent::SwmrRead, OpenIntent::Write] {
            assert!(allows(3, 0x02, intent), "{intent:?} refused flags 0x02");
        }
    }

    /// Versions below 3 are not checked at all: the C library raises the write
    /// bit on them and never reads it back, so refusing one would refuse a file
    /// `H5Fopen` accepts.
    #[test]
    fn an_older_superblock_is_not_checked() {
        for version in [0, 1, 2] {
            for intent in [READ, OpenIntent::SwmrRead, OpenIntent::Write] {
                assert!(
                    allows(version, 0x05, intent),
                    "v{version} superblock refused {intent:?} on flags 0x05"
                );
            }
        }
    }

    /// The refusal has to say what to do next: the path, the flags, and the
    /// recovery a user would otherwise have to find in the C library's docs.
    #[test]
    fn the_refusal_names_the_recovery() {
        let err = check_status_flags(&flagged(3, 0x05), READ, OpenTarget::Path(Path::new("d.h5")))
            .expect_err("a flagged file is refused for a snapshot read");
        let msg = err.to_string();
        assert!(matches!(err, Error::FileMarkedInUse(_)), "got {err:?}");
        // `from_bytes` is named because it is the only way through for a flagged
        // file on a read-only mount, where `clear_swmr_flag` cannot get the write
        // access it needs.
        for part in ["d.h5", "0x05", "clear_swmr_flag", "open_swmr", "from_bytes"] {
            assert!(msg.contains(part), "refusal does not mention {part}: {msg}");
        }
    }

    /// The recovery a source is offered is the one a source can reach.
    ///
    /// Two of the three the path refusal names — `open_swmr` to follow the
    /// writer, `clear_swmr_flag` to clear what one left behind — take a path,
    /// so handing them to a caller that opened a byte source would be advice it
    /// cannot take. The refusal has to say which half applies, and the check
    /// below is that it does: the path wording contains none of this text, so
    /// a `Source` target falling back to it fails here.
    #[test]
    fn a_source_is_told_which_recovery_it_can_reach() {
        let err = check_status_flags(&flagged(3, 0x05), READ, OpenTarget::Source)
            .expect_err("a flagged file is refused for a snapshot read");
        let msg = err.to_string();
        assert!(
            msg.starts_with("file is marked in use: a byte source:"),
            "the refusal does not name the source it refused: {msg}"
        );
        assert!(
            msg.contains("File::from_bytes to read the bytes as they stand"),
            "the refusal does not name the one recovery a source can reach: {msg}"
        );
        assert!(
            msg.contains("both need a filesystem path"),
            "the refusal offers path-only recoveries without saying they need a path: {msg}"
        );
    }

    /// The mark a page-buffered writer leaves is the write bit alone, which no
    /// SWMR reader can follow — so the refusal must not send the caller to
    /// `File::open_swmr`, which refuses the same file in its turn (issue #419).
    /// What applies instead is the opt-in, named in full so it can be pasted.
    #[test]
    fn a_write_only_mark_names_the_snapshot_opt_in() {
        let err = check_status_flags(
            &flagged(3, WRITE_ACCESS),
            READ,
            OpenTarget::Path(Path::new("buffered.h5")),
        )
        .expect_err("a marked file is refused for a snapshot read");
        let msg = err.to_string();
        assert!(matches!(err, Error::FileMarkedInUse(_)), "got {err:?}");
        for part in [
            "buffered.h5",
            "0x01",
            "FileAccessProperties::with_write_mark_policy(WriteMarkPolicy::AllowSnapshot)",
            "File::sync",
            "from_bytes",
            "clear_swmr_flag",
        ] {
            assert!(msg.contains(part), "refusal does not mention {part}: {msg}");
        }
        assert!(
            !msg.contains("open_swmr"),
            "refusal names a reader that refuses this mark too: {msg}"
        );
    }

    /// A source refused by the same mark is offered the opt-in, which it can
    /// take — it is a property of the open, not of the path — and told that the
    /// recovery it cannot take needs one.
    #[test]
    fn a_source_refused_by_a_write_only_mark_is_offered_the_opt_in() {
        let err = check_status_flags(&flagged(3, WRITE_ACCESS), READ, OpenTarget::Source)
            .expect_err("a marked file is refused for a snapshot read");
        let msg = err.to_string();
        assert!(
            msg.contains("with_write_mark_policy(WriteMarkPolicy::AllowSnapshot)"),
            "the refusal does not name the opt-in a source can take: {msg}"
        );
        assert!(
            msg.contains("File::clear_swmr_flag needs a filesystem path"),
            "the refusal offers a path-only recovery without saying it needs a path: {msg}"
        );
        assert!(!msg.contains("open_swmr"), "got {msg}");
    }

    /// The other half: the pair a SWMR writer leaves keeps the wording that
    /// names its reader, and does not offer an opt-in that would refuse it.
    #[test]
    fn a_swmr_pair_names_its_reader_and_not_the_opt_in() {
        let err = check_status_flags(
            &flagged(3, WRITE_ACCESS | SWMR_WRITE_ACCESS),
            READ,
            OpenTarget::Path(Path::new("swmr.h5")),
        )
        .expect_err("a marked file is refused for a snapshot read");
        let msg = err.to_string();
        assert!(msg.contains("File::open_swmr"), "got {msg}");
        assert!(
            !msg.contains("write_mark_policy"),
            "the opt-in does not admit a SWMR pair, so the refusal must not name it: {msg}"
        );
    }

    /// What the opt-in admits, stated as the whole rule: the write bit alone,
    /// and nothing carrying the SWMR bit — that file has a reader of its own.
    /// The read-write opens cannot reach this at all, since only
    /// [`OpenIntent::Read`] carries a policy.
    #[test]
    fn the_snapshot_opt_in_admits_the_write_mark_alone() {
        for flags in [0x00, WRITE_ACCESS, SWMR_WRITE_ACCESS, 0x05] {
            let allowed = flags & SWMR_WRITE_ACCESS == 0;
            assert_eq!(
                allows(3, flags, OpenIntent::Read(WriteMarkPolicy::AllowSnapshot)),
                allowed,
                "AllowSnapshot on flags {flags:#04x}"
            );
        }
    }

    /// Write a file at `path` whose superblock carries `version` and `flags`,
    /// re-serializing the superblock so its checksum stays valid.
    fn write_file_with(path: &Path, version: u8, flags: u32) {
        let mut bytes = crate::writer::FileBuilder::new().finish().unwrap();
        let off = crate::signature::find_signature(&bytes).unwrap();
        let mut sb = Superblock::parse(&bytes, off).unwrap();
        assert_eq!(sb.version, 3, "this writer emits a v3 superblock");
        sb.version = version;
        sb.consistency_flags = flags;
        let patched = sb.serialize();
        bytes[off..off + patched.len()].copy_from_slice(&patched);
        std::fs::write(path, &bytes).unwrap();
    }

    /// The version gate is checked against a real file, not only against a
    /// hand-built `Superblock`: a v2 file whose write flag is set still opens
    /// through `File::open`, which is the C-library parity the gate exists for.
    /// (v2 and v3 superblocks share a byte layout, so the rewrite above is the
    /// whole difference between them.)
    #[test]
    fn a_flagged_v2_file_still_opens() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v2.h5");
        write_file_with(&path, 2, WRITE_ACCESS | SWMR_WRITE_ACCESS);

        let file = crate::File::open(&path).expect("a v2 file's status flags are not checked");
        assert_eq!(file.superblock().consistency_flags, 0x05);
    }

    /// The other half of the version gate: nothing this crate does raises a flag
    /// on a superblock the gate skips, because the SWMR writer — the only path
    /// that raises one — refuses a pre-v3 file outright, as the C library does.
    #[test]
    fn the_swmr_writer_refuses_a_superblock_the_gate_would_skip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v2.h5");
        write_file_with(&path, 2, 0);

        let err = crate::File::open_swmr_writer(&path)
            .expect_err("SWMR writing requires a v3 superblock");
        assert!(
            matches!(err, Error::SwmrAppendUnsupported(_)),
            "got {err:?}"
        );
        let bytes = std::fs::read(&path).unwrap();
        let off = crate::signature::find_signature(&bytes).unwrap();
        assert_eq!(
            bytes[off + 11],
            0,
            "a refused writer must not have flagged the file on its way out"
        );
    }

    /// Half a flag pair is what a *plain* (non-SWMR) C-library writer leaves, and
    /// a SWMR reader has no protocol for following one, so `File::open_swmr`
    /// refuses it where it accepts the full pair. Exercised on a real file
    /// because the mismatch rule is the one branch a caller can reach only
    /// through a file another library wrote.
    #[test]
    fn a_swmr_reader_refuses_write_access_without_the_swmr_bit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("half.h5");
        write_file_with(&path, 3, WRITE_ACCESS);

        let err = crate::File::open_swmr(&path)
            .expect_err("a SWMR reader needs a SWMR writer, not a plain one");
        assert!(matches!(err, Error::FileMarkedInUse(_)), "got {err:?}");
        assert!(
            crate::File::open(&path).is_err(),
            "a snapshot read is refused too"
        );

        write_file_with(&path, 3, WRITE_ACCESS | SWMR_WRITE_ACCESS);
        crate::File::open_swmr(&path).expect("the full pair is the writer it follows");
    }
}
