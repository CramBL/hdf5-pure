//! Library-version bounds — the [`H5F_libver_t`] concept.
//!
//! Every HDF5 file is encoded with a set of *format versions* (superblock,
//! object headers, message encodings). A reader needs a library new enough to
//! understand those versions, and the HDF5 C API lets a writer bound which
//! versions a new file may use via `H5Pset_libver_bounds`. [`LibVer`] names the
//! release boundaries at which the on-disk format changed, so callers of this
//! crate can ask which format an existing file requires
//! ([`crate::File::libver_bound`]) or constrain what a new file may emit
//! ([`crate::FileBuilder::with_libver_bounds`]).
//!
//! [`H5F_libver_t`]: https://portal.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html

/// A library-version boundary, mirroring HDF5's `H5F_libver_t`.
///
/// Variants are ordered oldest to newest; a later variant understands strictly
/// more of the format than an earlier one. `LibVer` derives `Ord` on that
/// ordering, so bounds can be compared directly.
///
/// Non-exhaustive: the reference library keeps adding boundaries as the format
/// grows, so match with a `_` arm and read the newest this crate knows about
/// from [`LATEST`](Self::LATEST).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
// `mat::Options` carries a `LibVer` and is itself serializable, so a persisted
// set of MAT options can record which on-disk format it writes.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum LibVer {
    /// The earliest format (HDF5 1.0+): version 0/1 superblock, v1
    /// symbol-table groups. Readable by every released HDF5 library.
    Earliest,
    /// HDF5 1.8: version 2 superblock and the "new style" (version 2) object
    /// headers, dense link/attribute storage, and the v2 B-tree indices. The
    /// oldest format this crate's writer emits — see
    /// [`WRITER_OLDEST`](Self::WRITER_OLDEST).
    V18,
    /// HDF5 1.10: version 3 superblock, plus SWMR and the extensible/fixed
    /// array chunk indices. The format this crate's writer emits unless bounds
    /// say otherwise — see [`WRITER_DEFAULT`](Self::WRITER_DEFAULT).
    V110,
    /// HDF5 1.12.
    V112,
    /// HDF5 1.14.
    V114,
}

impl LibVer {
    /// The newest boundary this enum knows about — the meaning of
    /// `H5F_LIBVER_LATEST`. Tracks the highest concrete variant.
    ///
    /// Usable as either bound. As a *lower* bound it is a licence to use newer
    /// encodings, not a requirement to: `H5Pset_libver_bounds` lets the library
    /// pick anything from the low bound upward, and what this crate writes at
    /// [`WRITER_DEFAULT`](Self::WRITER_DEFAULT) is already no newer than the
    /// reference library emits under a low bound of 1.12 or 1.14, so a lower
    /// bound above `WRITER_DEFAULT` is satisfied by it rather than refused.
    pub const LATEST: LibVer = LibVer::V114;

    /// The on-disk format this crate's [`FileBuilder`](crate::FileBuilder)
    /// produces when nothing constrains it: the version 3 superblock introduced
    /// in HDF5 1.10.
    ///
    /// It is a *default*, not the only output.
    /// [`with_libver_bounds`](crate::FileBuilder::with_libver_bounds) selects a
    /// format within the bounds it is given, and the oldest this crate can write
    /// is [`WRITER_OLDEST`](Self::WRITER_OLDEST).
    pub const WRITER_DEFAULT: LibVer = LibVer::V110;

    /// The oldest on-disk format this crate's [`FileBuilder`](crate::FileBuilder)
    /// can produce: the version 2 superblock and "new style" object headers
    /// introduced in HDF5 1.8.
    ///
    /// Nothing older is reachable. A version 0 or 1 superblock pairs with v1
    /// symbol-table groups and local heaps, which this crate reads but does not
    /// write, so a bound whose upper end is [`Earliest`](Self::Earliest) is
    /// refused rather than silently satisfied with something newer.
    pub const WRITER_OLDEST: LibVer = LibVer::V18;

    /// The minimum library version required to read a file with the given
    /// superblock version — i.e. the *low bound* the on-disk format implies.
    ///
    /// Superblock 0/1 → [`Earliest`](LibVer::Earliest); 2 → [`V18`](LibVer::V18);
    /// 3 and anything newer → [`V110`](LibVer::V110).
    pub fn from_superblock_version(version: u8) -> LibVer {
        match version {
            0 | 1 => LibVer::Earliest,
            2 => LibVer::V18,
            _ => LibVer::V110,
        }
    }

    /// The format to write under `bounds`: the newest this crate produces that
    /// they admit, or [`WRITER_DEFAULT`](Self::WRITER_DEFAULT) when they impose
    /// nothing. Bounds admitting no such format give
    /// [`FormatError::LibverBoundsUnsatisfiable`](crate::FormatError::LibverBoundsUnsatisfiable).
    ///
    /// A lower bound above [`WRITER_DEFAULT`](Self::WRITER_DEFAULT) is satisfied
    /// by it, so `(LATEST, LATEST)` resolves to the 1.10 format rather than
    /// failing. HDF5's low bound is a licence to use newer encodings, not a
    /// requirement to use them, and every message this crate writes is already
    /// at or below the version the reference library emits under a low bound of
    /// 1.12 or 1.14. The value returned is the format actually written, so an
    /// error naming it stays true.
    ///
    /// That licence is not a licence to ignore `high`: an inverted range is
    /// refused, as `H5Pset_libver_bounds` refuses one, and the check has to come
    /// before the clamp that implements the licence — clamping first would turn
    /// `(LATEST, V110)` into a bound the 1.10 format satisfies.
    ///
    /// One function so the whole-file writer and an editing session's fapl
    /// answer the same bounds the same way; a second copy of this rule is how
    /// the two would come to disagree about which format a caller asked for.
    pub(crate) fn resolve_writable(
        bounds: Option<(LibVer, LibVer)>,
    ) -> Result<LibVer, crate::error::FormatError> {
        let Some((low, high)) = bounds else {
            return Ok(LibVer::WRITER_DEFAULT);
        };
        if low > high {
            return Err(crate::error::FormatError::LibverBoundsUnsatisfiable {
                writes: LibVer::WRITER_DEFAULT.name(),
                requested_low: low.name(),
                requested_high: high.name(),
            });
        }
        let effective_low = low.min(LibVer::WRITER_DEFAULT);
        for candidate in [LibVer::WRITER_DEFAULT, LibVer::WRITER_OLDEST] {
            if candidate >= effective_low && candidate <= high {
                return Ok(candidate);
            }
        }
        Err(crate::error::FormatError::LibverBoundsUnsatisfiable {
            writes: LibVer::WRITER_DEFAULT.name(),
            requested_low: low.name(),
            requested_high: high.name(),
        })
    }

    /// A short, stable label for diagnostics (e.g. error messages).
    pub fn name(self) -> &'static str {
        match self {
            LibVer::Earliest => "earliest",
            LibVer::V18 => "v1.8",
            LibVer::V110 => "v1.10",
            LibVer::V112 => "v1.12",
            LibVer::V114 => "v1.14",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LibVer;
    use crate::error::FormatError;

    /// Unbounded, and any bound admitting the 1.10 format, resolve to it.
    #[test]
    fn bounds_admitting_the_default_format_resolve_to_it() {
        assert_eq!(LibVer::resolve_writable(None).unwrap(), LibVer::V110);
        for (low, high) in [
            (LibVer::Earliest, LibVer::LATEST),
            (LibVer::Earliest, LibVer::V110),
            (LibVer::V110, LibVer::V110),
        ] {
            assert_eq!(
                LibVer::resolve_writable(Some((low, high))).unwrap(),
                LibVer::V110,
                "bounds [{}, {}]",
                low.name(),
                high.name()
            );
        }
    }

    /// A lower bound newer than the 1.10 format this crate writes is a licence to
    /// use newer encodings, not a requirement to, so it is satisfied by that
    /// format — and resolves to the format written rather than the bound asked
    /// for.
    #[test]
    fn a_lower_bound_above_the_default_format_is_satisfied_by_it() {
        for (low, high) in [
            (LibVer::LATEST, LibVer::LATEST),
            (LibVer::V112, LibVer::LATEST),
            (LibVer::V112, LibVer::V112),
            (LibVer::V114, LibVer::V114),
        ] {
            assert_eq!(
                LibVer::resolve_writable(Some((low, high))).unwrap(),
                LibVer::V110,
                "bounds [{}, {}]",
                low.name(),
                high.name()
            );
        }
    }

    /// An upper bound of 1.8 selects the older format, since `high` is what picks
    /// between the two. Only a lower bound the 1.8 format satisfies gets there:
    /// `(V110, V18)` is inverted and refused.
    #[test]
    fn an_upper_bound_of_1_8_selects_the_older_format() {
        for (low, high) in [(LibVer::Earliest, LibVer::V18), (LibVer::V18, LibVer::V18)] {
            assert_eq!(
                LibVer::resolve_writable(Some((low, high))).unwrap(),
                LibVer::V18,
                "bounds [{}, {}]",
                low.name(),
                high.name()
            );
        }
    }

    /// What is left unsatisfiable: an upper bound older than the 1.8 format, and
    /// an inverted range whose upper bound sits below the lower one.
    ///
    /// The inverted pairs here all carry a `high` of 1.10 or newer, so the only
    /// thing that can refuse them is the inversion itself. `(LATEST, V18)` would
    /// pass this test with the inversion check gone, since a `high` of 1.8 is
    /// below the format the writer defaults to and is refused on that ground
    /// alone.
    #[test]
    fn bounds_admitting_no_format_this_crate_writes_are_refused() {
        for (low, high) in [
            (LibVer::Earliest, LibVer::Earliest),
            (LibVer::LATEST, LibVer::V18),
            (LibVer::V114, LibVer::V110),
            (LibVer::LATEST, LibVer::V112),
            (LibVer::V112, LibVer::V110),
            (LibVer::V110, LibVer::V18),
        ] {
            let err = LibVer::resolve_writable(Some((low, high))).unwrap_err();
            assert!(
                matches!(
                    err,
                    FormatError::LibverBoundsUnsatisfiable {
                        writes: "v1.10",
                        ..
                    }
                ),
                "bounds [{}, {}] gave {err:?}",
                low.name(),
                high.name()
            );
        }
    }
}
