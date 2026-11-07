//! Library-version boundaries for output format selection.
//!
//! [`LibVer::WRITER_DEFAULT`] is the default superblock and layout profile. Datatypes can
//! require a newer library than these structures. During file creation, the writer constrains
//! emitted content according to the bounds configured through
//! [`FileBuilder::with_libver_bounds`](crate::FileBuilder::with_libver_bounds).

use crate::SuperblockVersion;
use crate::datatype::Datatype;
use crate::error;

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
    /// array chunk indices. The writer's default superblock and layout profile is
    /// [`WRITER_DEFAULT`](Self::WRITER_DEFAULT).
    V110,
    /// HDF5 1.12.
    V112,
    /// HDF5 1.14.
    V114,
}

impl LibVer {
    /// The newest library-version boundary represented by this enum.
    ///
    /// When both bounds are set to this value, the writer uses
    /// [`WRITER_DEFAULT`](Self::WRITER_DEFAULT) for the superblock and layout profile. The writer
    /// rejects native complex datatypes during file creation under this explicit upper bound.
    pub const LATEST: LibVer = LibVer::V114;

    /// The default superblock and layout profile, introduced in HDF5 1.10.
    ///
    /// [`FileBuilder`](crate::FileBuilder) writes a version 3 superblock under this profile.
    /// Emitted datatypes can require a newer library version. Without explicit bounds, the
    /// writer permits native complex datatypes, which require HDF5 2.0.
    /// [`FileBuilder::with_libver_bounds`](crate::FileBuilder::with_libver_bounds) selects a
    /// profile and constrains emitted content. The oldest supported profile is
    /// [`WRITER_OLDEST`](Self::WRITER_OLDEST).
    pub const WRITER_DEFAULT: LibVer = LibVer::V110;

    /// The oldest superblock and layout profile the writer supports.
    ///
    /// Under this HDF5 1.8 profile, the writer uses a version 2 superblock and version 3
    /// layout messages. An upper bound of [`Earliest`](Self::Earliest) is rejected with
    /// [`FormatError::LibverBoundsUnsatisfiable`](crate::FormatError::LibverBoundsUnsatisfiable).
    pub const WRITER_OLDEST: LibVer = LibVer::V18;

    /// Returns the minimum library version required by a superblock version.
    ///
    /// Returns [`Earliest`](Self::Earliest) for versions 0 and 1, [`V18`](Self::V18) for version 2,
    /// and [`V110`](Self::V110) for version 3.
    pub fn from_superblock_version(version: SuperblockVersion) -> LibVer {
        match version {
            SuperblockVersion::Zero | SuperblockVersion::One => LibVer::Earliest,
            SuperblockVersion::Two => LibVer::V18,
            SuperblockVersion::Three => LibVer::V110,
        }
    }

    /// Returns the newest supported superblock and layout profile permitted by `bounds`.
    ///
    /// Uses [`WRITER_DEFAULT`](Self::WRITER_DEFAULT) when bounds are omitted or when the lower
    /// bound exceeds that profile. [`WritableTarget::with_datatypes`] checks datatype requirements
    /// against the explicit upper bound.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`FormatError::LibverBoundsUnsatisfiable`](crate::FormatError::LibverBoundsUnsatisfiable)
    /// if the bounds are inverted or the upper bound is [`Earliest`](Self::Earliest).
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

/// A parsed output target with a superblock and layout profile and an optional feature ceiling.
///
/// The parser retains the explicit upper bound separately from the selected profile. The writer
/// adds datatype requirements through [`Self::with_datatypes`] before emission.
#[derive(Clone, Copy)]
pub(crate) struct WritableTarget {
    profile: LibVer,
    ceiling: Option<LibVer>,
    content: ContentVersion,
}

impl WritableTarget {
    /// Parses bounds and selects the superblock and layout profile.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`FormatError::LibverBoundsUnsatisfiable`](crate::FormatError::LibverBoundsUnsatisfiable)
    /// if the bounds are inverted or their upper bound is [`LibVer::Earliest`].
    pub(crate) fn parse(bounds: Option<(LibVer, LibVer)>) -> Result<Self, error::FormatError> {
        Ok(Self {
            profile: LibVer::resolve_writable(bounds)?,
            ceiling: bounds.map(|(_, high)| high),
            content: ContentVersion::V18,
        })
    }

    /// Adds recursive datatype requirements to this target.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::LibverTooOldForContent`](crate::FormatError::LibverTooOldForContent)
    /// if bounds were supplied and a datatype contains native complex content.
    pub(crate) fn with_datatypes<'a>(
        mut self,
        datatypes: impl IntoIterator<Item = &'a Datatype>,
    ) -> Result<Self, error::FormatError> {
        for datatype in datatypes {
            self.content = self.content.max(ContentVersion::from_datatype(datatype));
        }
        if let Some(ceiling) = self.ceiling {
            match self.content {
                ContentVersion::V18 => {}
                ContentVersion::V200 => {
                    return Err(error::FormatError::LibverTooOldForContent {
                        content: "a native complex datatype",
                        needs: "v2.0",
                        writing: ceiling.name(),
                    });
                }
            }
        }
        Ok(self)
    }

    /// Returns the profile used for superblock and layout versions.
    pub(crate) fn profile(self) -> LibVer {
        self.profile
    }
}

/// The minimum output release tracked for datatype compatibility.
#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum ContentVersion {
    V18,
    // H5Tcomplex_create introduces native complex in H5Tcomplex.c, HDF5 2.0.0.
    V200,
}

impl ContentVersion {
    fn from_datatype(datatype: &Datatype) -> Self {
        match datatype {
            Datatype::Complex(_) => Self::V200,
            Datatype::Array { base_type, .. } | Datatype::Enumeration { base_type, .. } => {
                Self::from_datatype(base_type)
            }
            Datatype::Compound { members, .. } => members
                .iter()
                .map(|member| Self::from_datatype(&member.datatype))
                .max()
                .unwrap_or(Self::V18),
            Datatype::VariableLength(kind) => Self::from_datatype(kind.base_type()),
            _ => Self::V18,
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

    #[rstest::rstest]
    #[case::earliest_only(LibVer::Earliest, LibVer::Earliest, "earliest", "earliest")]
    #[case::latest_below_v18(LibVer::LATEST, LibVer::V18, "v1.14", "v1.8")]
    #[case::v114_below_v110(LibVer::V114, LibVer::V110, "v1.14", "v1.10")]
    #[case::latest_below_v112(LibVer::LATEST, LibVer::V112, "v1.14", "v1.12")]
    #[case::v112_below_v110(LibVer::V112, LibVer::V110, "v1.12", "v1.10")]
    #[case::v110_below_v18(LibVer::V110, LibVer::V18, "v1.10", "v1.8")]
    fn bounds_admitting_no_format_this_crate_writes_are_rejected(
        #[case] low: LibVer,
        #[case] high: LibVer,
        #[case] requested_low: &'static str,
        #[case] requested_high: &'static str,
    ) {
        assert_eq!(
            LibVer::resolve_writable(Some((low, high))).unwrap_err(),
            FormatError::LibverBoundsUnsatisfiable {
                writes: "v1.10",
                requested_low,
                requested_high,
            }
        );
    }
}
