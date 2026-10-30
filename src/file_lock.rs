//! OS file locking and superblock status flags for filesystem opens.
//!
//! [`acquire_exclusive`] attempts to acquire an exclusive OS lock. The OS releases a lock when
//! the process exits. [`check_status_flags`] checks the durable superblock flags, which can
//! remain set after a writer exits without closing the file.
//!
//! With locking enabled, editors acquire an exclusive lock. Readers and SWMR writers run without a
//! lock so that a
//! SWMR writer can publish appends to concurrent readers, including on Windows, where a file
//! lock blocks reads through other handles.
//!
//! [`crate::File::clear_status_flags`] clears flags left by an interrupted ordinary or SWMR
//! writer, including a page-buffered session. The caller must ensure that the writer has
//! stopped, since the exclusive-lock attempt cannot exclude a live SWMR writer.

use std::fs::{File, TryLockError};
use std::path::Path;

use crate::SuperblockStatusFlag;
use crate::SuperblockStatusFlags;
use crate::SuperblockVersion;
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
             crashed, the OS lock is released automatically (try again); leftover \
             on-disk status flags can be cleared with File::clear_status_flags. Set \
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
/// for write access. libhdf5 raises it for *any* writer. This
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
pub(crate) const WRITE_ACCESS: SuperblockStatusFlag = SuperblockStatusFlag::WriteAccess;

/// Superblock status-flag bit 2 (`H5F_SUPER_SWMR_WRITE_ACCESS`): the writer
/// holding the file is a SWMR writer, so a SWMR reader may attach to it.
pub(crate) const SWMR_WRITE_ACCESS: SuperblockStatusFlag = SuperblockStatusFlag::SwmrWriteAccess;

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
/// [`crate::File::clear_status_flags`], need a filesystem path. Naming them to a
/// caller who opened a byte source would be advice it cannot take.
///
/// A closed set rather than a `&dyn Display`, so the wording every refusal
/// leads with lives here beside the reasons it is concatenated with, and a
/// third kind of open arrives as a variant a reviewer sees.
#[derive(Debug, Clone, Copy)]
pub(crate) enum OpenTarget<'a> {
    /// A file opened by name, which the refusal reports and whose flag
    /// [`crate::File::clear_status_flags`] can clear in place.
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
    /// `clear_status_flags` to write to. The mark, because `open_swmr` follows a
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
                 the flag with File::clear_status_flags"
            }
            (Self::Source, true) => {
                "Use File::from_bytes to read the bytes as they stand; following a live writer \
                 with File::open_swmr, and clearing a flag a writer left behind with \
                 File::clear_status_flags, both need a filesystem path"
            }
            (Self::Path(_), false) => {
                "No SWMR writer holds it, so no reader can follow it as one. Pass \
                 FileAccessProperties::with_write_mark_policy(WriteMarkPolicy::AllowSnapshot) to \
                 read it as it stands, which is a consistent snapshot once the writer has called \
                 File::sync or closed the file, or use File::from_bytes to read the bytes as they \
                 stand; if a writer exited without closing the file, clear the flag with \
                 File::clear_status_flags"
            }
            (Self::Source, false) => {
                "No SWMR writer holds it, so no reader can follow it as one. Pass \
                 FileAccessProperties::with_write_mark_policy(WriteMarkPolicy::AllowSnapshot) to \
                 read it as it stands, which is a consistent snapshot once the writer has called \
                 File::sync or closed the file, or use File::from_bytes to read the bytes as they \
                 stand; clearing a flag a writer left behind with File::clear_status_flags needs a \
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
/// error reports [`crate::File::clear_status_flags`] as the recovery, without
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
    if superblock.version() != SuperblockVersion::Three {
        return Ok(());
    }
    let flags = superblock.consistency_flags();
    let write = flags.contains(WRITE_ACCESS);
    let swmr = flags.contains(SWMR_WRITE_ACCESS);
    // Each arm states the whole condition: "marked open for write" describes two
    // of the three, and reporting the SWMR-read mismatch that way would misname
    // the case where only the SWMR bit is set.
    let reason = match intent {
        OpenIntent::Write if write || swmr => format!(
            "the superblock marks the file as open for write (status flags {flags:#04x}), so \
             another writer holds it. Open it read-only, or — if a writer exited without \
             closing the file — clear the flag with File::clear_status_flags",
            flags = flags.bits(),
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
                target.read_recovery(swmr),
                flags = flags.bits(),
            )
        }
        OpenIntent::SwmrRead if write != swmr => format!(
            "the superblock's status flags disagree ({flags:#04x}): a SWMR reader needs a SWMR \
             writer (both the write and SWMR-write bits) or a quiescent file (neither). Clear \
             them with File::clear_status_flags if a writer exited without closing the file",
            flags = flags.bits(),
        ),
        _ => return Ok(()),
    };
    Err(Error::FileMarkedInUse(format!("{target}: {reason}.")))
}

/// Clears the status flags in `path` and updates the superblock checksum.
///
/// Version 0 and 1 superblocks are left unchanged. [`check_status_flags`] checks version 3,
/// and the serializer supports versions 2 and 3.
///
/// # Errors
///
/// Returns [`Error::FileLocked`], [`Error::Io`], or [`Error::Format`] under the conditions
/// documented by [`crate::File::clear_status_flags`].
pub(crate) fn clear_status_flags_at(path: &Path) -> Result<(), Error> {
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom, Write};

    use crate::superblock;
    use crate::superblock::LocatedSuperblock;

    let mut w = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(Error::Io)?;
    // With locking enabled, a conflicting lock prevents clearing flags. A SWMR writer runs
    // without a lock, so the caller must ensure that it has stopped.
    acquire_exclusive(&w, FileLocking::Enabled, path)?;
    let mut data = Vec::new();
    w.read_to_end(&mut data).map_err(Error::Io)?;
    let LocatedSuperblock {
        address,
        superblock: mut sb,
        ..
    } = superblock::read(data.as_slice())?;
    if matches!(
        sb.version(),
        SuperblockVersion::Zero | SuperblockVersion::One
    ) {
        // `serialize_superblock` writes only the v2/v3 layout and returns an error for a v0/v1
        // superblock. `check_status_flags`, like `H5Fopen`, reads the status flags from version 3
        // on, so a flag left in a v0/v1 superblock does not block an open.
        return Ok(());
    }
    if sb.consistency_flags().is_empty() {
        return Ok(());
    }
    sb.set_consistency_flags(SuperblockStatusFlags::new())?;
    let bytes = hdf5_pure_format::__private::serialize_superblock(&sb)?;
    w.seek(SeekFrom::Start(address.get())).map_err(Error::Io)?;
    w.write_all(&bytes).map_err(Error::Io)?;
    w.sync_data().map_err(Error::Io)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use hdf5_pure_core::__private::SuperblockFields;
    use test_util_hdf5::assertion::StatusOpen;

    use super::*;
    use crate::SuperblockStatus;
    use crate::address::BaseAddressExt;
    use crate::address::StoredAddress;
    use crate::address::StoredAddressExt;
    use crate::superblock::LocatedSuperblock;
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
    fn flagged(version: SuperblockVersion, flags: SuperblockStatusFlags) -> Superblock {
        SuperblockFields {
            status: SuperblockStatus::new(version, flags).unwrap(),
            offset_size: OffsetWidth::Eight,
            length_size: LengthWidth::Eight,
            base_address: crate::address::BaseAddress::ZERO,
            eof_address: 0,
            root_group_address: StoredAddress::ZERO,
            group_leaf_node_k: None,
            group_internal_node_k: None,
            indexed_storage_internal_node_k: None,
            free_space_address: None,
            driver_info_address: None,
            superblock_extension_address: None,
            checksum: None,
        }
        .build()
    }

    /// A snapshot read under the default policy, which is what every rule
    /// below is stated against unless it names the opt-in.
    const READ: OpenIntent = OpenIntent::Read(WriteMarkPolicy::Refuse);

    #[rstest::rstest]
    #[case::unmarked(0x00, [true, true, true, true])]
    #[case::write_mark(0x01, [false, false, true, false])]
    #[case::swmr_mark_without_write(0x04, [false, false, false, false])]
    #[case::write_and_swmr_marks(0x05, [false, false, false, true])]
    #[case::unrelated_low_bit(0x02, [true, true, true, true])]
    #[case::write_mark_with_high_bit(0x81, [false, false, true, false])]
    #[case::swmr_mark_with_high_bit(0x84, [false, false, false, false])]
    #[case::all_bits(0xFF, [false, false, false, true])]
    fn status_flags_follow_the_version_and_open_intent(
        #[case] bits: u32,
        #[case] allowed: [bool; 4],
        #[values(
            SuperblockVersion::Zero,
            SuperblockVersion::One,
            SuperblockVersion::Two,
            SuperblockVersion::Three
        )]
        version: SuperblockVersion,
        #[values((OpenIntent::Write, 0), (READ, 1), (OpenIntent::Read(WriteMarkPolicy::AllowSnapshot), 2), (OpenIntent::SwmrRead, 3))]
        (intent, index): (OpenIntent, usize),
    ) {
        let flags = hdf5_pure_core::__private::superblock_status_flags(bits);
        let result = check_status_flags(&flagged(version, flags), intent, OpenTarget::Source);
        if version != SuperblockVersion::Three || allowed[index] {
            result.unwrap();
        } else {
            assert_source_status_error(result.unwrap_err(), flags, intent);
        }
    }

    #[rstest::rstest]
    #[case::swmr_path_d(
        SuperblockStatusFlags::swmr_write_marked(),
        OpenTarget::Path(Path::new("d.h5"))
    )]
    #[case::swmr_source(SuperblockStatusFlags::swmr_write_marked(), OpenTarget::Source)]
    #[case::write_mark_path(
        SuperblockStatusFlags::write_marked(),
        OpenTarget::Path(Path::new("buffered.h5"))
    )]
    #[case::write_mark_source(SuperblockStatusFlags::write_marked(), OpenTarget::Source)]
    #[case::swmr_path_swmr(
        SuperblockStatusFlags::swmr_write_marked(),
        OpenTarget::Path(Path::new("swmr.h5"))
    )]
    fn a_marked_read_reports_complete_recovery(
        #[case] flags: SuperblockStatusFlags,
        #[case] target: OpenTarget<'_>,
    ) {
        let error = check_status_flags(&flagged(SuperblockVersion::Three, flags), READ, target)
            .unwrap_err();
        let open = match target {
            OpenTarget::Path(path) => StatusOpen::Read(path),
            OpenTarget::Source => StatusOpen::SourceRead,
        };
        assert_status_error(error, flags, open);
    }

    /// Write a file at `path` whose superblock carries `version` and `flags`,
    /// re-serializing the superblock so its checksum stays valid.
    fn write_file_with(path: &Path, version: SuperblockVersion, flags: SuperblockStatusFlags) {
        let mut bytes = crate::writer::FileBuilder::new().finish().unwrap();
        let LocatedSuperblock {
            address,
            superblock: mut sb,
            ..
        } = crate::superblock::read(bytes.as_slice()).unwrap();
        let off = usize::try_from(address.get()).unwrap();
        assert_eq!(
            sb.version(),
            SuperblockVersion::Three,
            "this writer emits a v3 superblock"
        );
        sb.set_version(version).unwrap();
        sb.set_consistency_flags(flags).unwrap();
        let patched = hdf5_pure_format::__private::serialize_superblock(&sb).unwrap();
        bytes[off..off + patched.len()].copy_from_slice(&patched);
        std::fs::write(path, &bytes).unwrap();
    }

    #[test]
    fn a_flagged_v2_file_still_opens() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v2.h5");
        write_file_with(
            &path,
            SuperblockVersion::Two,
            SuperblockStatusFlags::swmr_write_marked(),
        );

        let file = crate::File::open(&path).expect("a v2 file's status flags are not checked");
        assert_eq!(
            file.superblock().consistency_flags(),
            SuperblockStatusFlags::swmr_write_marked()
        );
    }

    #[test]
    fn the_swmr_writer_rejects_a_superblock_the_gate_skips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v2.h5");
        write_file_with(&path, SuperblockVersion::Two, SuperblockStatusFlags::new());

        let err = crate::File::open_swmr_writer(&path)
            .expect_err("SWMR writing requires a v3 superblock");
        assert!(
            matches!(
                err,
                Error::SwmrAppendUnsupported(
                    "SWMR writing requires a latest-format file (v3 superblock) with no userblock and no persisted free-space"
                )
            ),
            "{err:?}"
        );
        let bytes = std::fs::read(&path).unwrap();
        let off = usize::try_from(
            crate::superblock::read(bytes.as_slice())
                .unwrap()
                .address
                .get(),
        )
        .unwrap();
        assert_eq!(
            bytes[off + 11],
            0,
            "a refused writer must not have flagged the file on its way out"
        );
    }

    #[test]
    fn a_swmr_reader_rejects_write_access_without_the_swmr_bit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("half.h5");
        write_file_with(
            &path,
            SuperblockVersion::Three,
            SuperblockStatusFlags::write_marked(),
        );

        let err = crate::File::open_swmr(&path)
            .expect_err("a SWMR reader needs a SWMR writer, not a plain one");
        assert_status_error(
            err,
            SuperblockStatusFlags::write_marked(),
            StatusOpen::SwmrRead(&path),
        );
        assert_status_error(
            crate::File::open(&path).unwrap_err(),
            SuperblockStatusFlags::write_marked(),
            StatusOpen::Read(&path),
        );

        write_file_with(
            &path,
            SuperblockVersion::Three,
            SuperblockStatusFlags::swmr_write_marked(),
        );
        let file = crate::File::open_swmr(&path).unwrap();
        assert_eq!(file.root().datasets().unwrap(), Vec::<String>::new());
    }
    fn assert_source_status_error(error: Error, flags: SuperblockStatusFlags, intent: OpenIntent) {
        let bits = flags.bits();
        let reason = match intent {
            OpenIntent::Read(_) => {
                assert_status_error(error, flags, StatusOpen::SourceRead);
                return;
            }
            OpenIntent::Write => format!(
                "a byte source: the superblock marks the file as open for write (status flags {bits:#04x}), so another writer holds it. Open it read-only, or — if a writer exited without closing the file — clear the flag with File::clear_status_flags."
            ),
            OpenIntent::SwmrRead => format!(
                "a byte source: the superblock's status flags disagree ({bits:#04x}): a SWMR reader needs a SWMR writer (both the write and SWMR-write bits) or a quiescent file (neither). Clear them with File::clear_status_flags if a writer exited without closing the file."
            ),
        };
        let Error::FileMarkedInUse(actual) = error else {
            panic!("expected FileMarkedInUse, got {error:?}");
        };
        assert_eq!(actual, reason);
    }
    fn assert_status_error(error: Error, flags: SuperblockStatusFlags, open: StatusOpen<'_>) {
        let printed = error.to_string();
        let Error::FileMarkedInUse(actual) = error else {
            panic!("expected FileMarkedInUse, got {error:?}");
        };
        let expected = open.message(flags);
        assert_eq!(actual, expected);
        assert_eq!(printed, format!("file is marked in use: {expected}"));
    }
}
