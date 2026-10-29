//! Configuration knobs for MAT v7.3 writers.
//!
//! [`Options`] groups every policy a MATLAB writer might need to flex on:
//! string class (`char` vs `string`), 1-D vector orientation, name validation
//! behavior, compression, and more.
//!
//! The defaults match the historical hdf5-pure serde writer (`String -> char`,
//! no name sanitization, no compression, column vectors). Callers wanting the
//! richer "modern MATLAB" output should construct an `Options` with explicit
//! `string_class: StringClass::String`, `invalid_name_policy: Sanitize`, etc.

#[cfg(feature = "serde")]
use serde::Deserializer;
#[cfg(feature = "serde")]
use serde::Serializer;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::DeflateLevel;
use crate::libver::LibVer;

/// MATLAB class to use when emitting Rust `String` (or BEVE string) values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum StringClass {
    /// Encode as `char` (UTF-16 row vector). Compatible with `strcmp`, but
    /// not with the `==` operator.
    Char,
    /// Encode as the modern MATLAB `string` class via `mxOPAQUE_CLASS`.
    /// Supports `==` semantics. Costs a `#refs#` payload and a
    /// `#subsystem#/MCOS` entry per writer.
    String,
}

/// Encoding for 1-D vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum OneDimensionalMode {
    /// MATLAB shape `[N, 1]`.
    ColumnVector,
    /// MATLAB shape `[1, N]`.
    RowVector,
}

/// Behavior for a `null` / `Option::None` in a struct field, a map value, or a
/// sequence element.
///
/// The file root is not one of those: it names no slot, so it is the variable
/// namespace rather than a value in one. A root `None` / `()` / unit struct
/// writes a valid MAT file with no variables under every policy except
/// [`Error`](Self::Error), which refuses it. That is the same file an empty root
/// map or a fieldless struct produces, and the reference library both writes and
/// reads one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum NullPolicy {
    /// Map `None` to MATLAB `struct([])` (an empty struct array). The field
    /// stays present, so MATLAB code can reference it unconditionally and test
    /// it with `isempty(fieldnames(x))`, which is what this crate's MATLAB
    /// fixture script checks. The two forms still differ for a struct with no
    /// fields, and only the `fieldnames` form has been verified against MATLAB
    /// itself.
    ///
    /// A bare `isempty(x)` became reliable in 0.34, when
    /// [`EmptyMarkerEncoding::DataAsDims`] became the default: the marker now
    /// carries its own dimensions, so the reference library recovers `0x0` with
    /// zero elements. Under the previous encoding there were no dimensions to
    /// recover, and the element count came back as one.
    ///
    /// Not expressible at the root, where `struct([])` would need a variable
    /// name to hang on; see the note on this enum.
    ///
    /// Reading such a field back is lenient but not universal. It deserializes
    /// into `Option<T>` as `None`, `Vec<T>` as empty, `serde_json::Value` as
    /// `Null`, and `()` as `()`. It does *not* deserialize into a bare scalar,
    /// `String`, struct or map: those report a type error, and `#[serde(default)]`
    /// does not help, because the field is present rather than missing. A reader
    /// that wants a default for a null field needs an `Option<T>` field.
    EmptyStructArray,
    /// Drop the field from its parent struct entirely, so `isfield` reports
    /// `false` and `s.field` raises. This is what the serde writer did
    /// unconditionally before 0.30.
    ///
    /// A sequence element is not droppable the way a field is: a cell array's
    /// element count is fixed by its dims, so a null element takes the
    /// `struct([])` marker instead.
    ///
    /// At the root this yields a valid file with zero variables. Note that it
    /// does not read back as `None`: the deserializer presents the root as a
    /// struct, so `from_bytes::<Option<T>>` on such a file fails, or yields
    /// `Some` of an all-defaulted `T`. Round-trip a root-level `None` as a field
    /// of a wrapper struct instead.
    Omit,
    /// Reject `None` with an error, including at the root.
    Error,
}

/// Encoding for a fieldless (unit) enum variant.
///
/// Only the serde *writer* consults this: serde hands the serializer both the
/// variant's index and its name, so either encoding is reachable. The BEVE
/// walker has no such choice, since BEVE records whichever one the producing
/// encoder chose. Reading is not conditioned on it either — the deserializer
/// resolves a variant from a name or from an index whichever way the file was
/// written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum UnitVariantEncoding {
    /// Emit the variant's name, honoring `#[serde(rename)]`. Self-describing
    /// in MATLAB, and stable across a reordering of the enum.
    Name,
    /// Emit the variant's zero-based declaration index as a `uint32`.
    ///
    /// This is the index serde reports, which counts variants from zero and
    /// ignores any explicit discriminant: `enum E { A = 5 }` writes `0`.
    ///
    /// Prefer [`Name`](Self::Name) unless a reader already expects the
    /// integer. An index means nothing without the schema that fixes the
    /// ordering, so inserting or reordering a variant silently changes what
    /// existing files decode to, and in MATLAB it reads as a magic number
    /// rather than something `strcmp` can check. This exists for files whose
    /// layout is already fixed, not as a default worth choosing.
    Index,
}

/// Encoding for a sequence that turned out to have no elements.
///
/// Only the serde writer consults this. An empty `serialize_seq` carries no
/// element type, so there is nothing to infer a MATLAB class from; the BEVE
/// walker does not need the hint, because a BEVE typed array names its element
/// type even when the array is empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum EmptySequencePolicy {
    /// Emit an empty `double` array, MATLAB's universal `[]`.
    DoubleArray,
    /// Emit an empty cell array, `{}`. Right when the sequence would have held
    /// structs or mixed types had it been populated.
    Cell,
}

/// Behavior when a Rust struct field name or BEVE key is not a valid MATLAB
/// identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum InvalidNamePolicy {
    /// Return an error with the offending name.
    Error,
    /// Rewrite the name into a valid identifier and deduplicate.
    Sanitize,
}

/// Behavior for BEVE values that have no direct MATLAB encoding (bf16, f16,
/// 128-bit integers, unknown extensions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum UnsupportedPolicy {
    /// Reject unsupported values with an error.
    Error,
    /// Convert unsupported scalar values to their string representation and
    /// emit them as MATLAB `string` objects.
    StringFallback,
    /// Widen low-precision floats (bf16, f16) to MATLAB `single`.
    LossyNumericWidening,
}

/// HDF5 dataset compression settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum Compression {
    /// No compression (no chunking).
    None,
    /// Deflate compression with an optional shuffle filter.
    ///
    /// With `serde`, the level is serialized as a number. Deserialization rejects values
    /// outside `0..=9`.
    Deflate {
        /// The level passed to the compressor.
        #[cfg_attr(
            feature = "serde",
            serde(
                serialize_with = "serialize_deflate_level",
                deserialize_with = "deserialize_deflate_level"
            )
        )]
        level: DeflateLevel,
        /// Apply HDF5 byte-shuffle filter before deflate.
        shuffle: bool,
    },
}

/// Behavior for row-major matrix payloads (e.g. BEVE `MatrixLayout::Right`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum RowMajorPolicy {
    /// Reorder row-major payloads into MATLAB column-major layout.
    ReorderToColumnMajor,
    /// Return an error rather than reordering.
    Error,
}

/// Marker encoding to use for empty values (`Vec<T>` of length 0,
/// `struct([])`, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum EmptyMarkerEncoding {
    /// Zero-element dataset of shape `[0, 0]` with `MATLAB_empty=1`.
    /// (hdf5-pure 0.3 historical default.)
    ZeroElement,
    /// One-element-per-dim `uint64` dataset whose payload is the dimension
    /// vector, with `MATLAB_empty=1`. (beve historical default.)
    DataAsDims,
}

/// Aggregated options for MAT v7.3 writers.
///
/// Two writers consume this: the serde writer in this crate, and the BEVE
/// walker that downstream crates build on [`MatBuilder`](crate::mat::MatBuilder).
/// Most knobs apply to both. The ones that cannot are marked in their field
/// docs, and they are asymmetric only because the two input formats carry
/// different type information, never as an oversight.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
// Applied to the struct, not to individual fields: this type gains fields, and a
// persisted `Options` written by an older version must keep loading. Per-field
// attributes would have to be remembered on every future addition, which is
// exactly what was forgotten when `unit_variant_encoding` and
// `empty_sequence_policy` were added.
#[cfg_attr(feature = "serde", serde(default))]
#[non_exhaustive]
pub struct Options {
    /// MATLAB class for string values.
    pub string_class: StringClass,
    /// HDF5 dataset compression settings.
    pub compression: Compression,
    /// Policy for invalid MATLAB names.
    pub invalid_name_policy: InvalidNamePolicy,
    /// Policy for `null` / `Option::None`.
    pub null_policy: NullPolicy,
    /// Encoding for fieldless enum variants. Serde writer only.
    pub unit_variant_encoding: UnitVariantEncoding,
    /// Encoding for sequences that turned out to be empty. Serde writer only.
    pub empty_sequence_policy: EmptySequencePolicy,
    /// Policy for unsupported numeric/BEVE types.
    pub unsupported_policy: UnsupportedPolicy,
    /// 1-D vector orientation.
    pub one_dimensional_mode: OneDimensionalMode,
    /// Row-major matrix payload handling.
    pub row_major_policy: RowMajorPolicy,
    /// Empty marker encoding.
    pub empty_marker_encoding: EmptyMarkerEncoding,
    /// The newest HDF5 on-disk format the file may use — the upper bound handed
    /// to [`FileBuilder::with_libver_bounds`](crate::FileBuilder::with_libver_bounds).
    ///
    /// Defaults to [`LibVer::V18`], the newest format every MATLAB release can
    /// open. A version 3 superblock is an HDF5 1.10 addition, and MATLAB linked
    /// HDF5 1.8.12 before R2021b (1.10.7 in R2021b, 1.10.x through R2024a,
    /// 1.14.4.3 since R2024b), so a file carrying one cannot be opened there at
    /// all. Around R2021b MathWorks additionally kept 1.8.12 on the MAT v7.3
    /// path while `h5read`/`h5disp` used 1.10.7, which is why such a file could
    /// print under `h5disp` and still fail to `load`.
    ///
    /// Do not read the version table above as saying which formats `load`
    /// accepts. R2023a reports HDF5 1.10.8 — a library that reads a version 3
    /// superblock without difficulty — and its `load` refuses one anyway, with
    /// "Not a binary MAT-file". The 1.8 default is what a release the table
    /// would call safe still requires.
    ///
    /// Real MATLAB writes an older format still — a version 0 superblock with
    /// v1 symbol-table groups, which this crate reads but does not produce.
    ///
    /// Raising this to [`LibVer::V110`] is what [`Compression`] needs, since
    /// compression requires chunked storage and the chunk indices this crate
    /// writes arrived in 1.10. The two are refused together rather than
    /// silently resolved, so a file that cannot be loaded by MATLAB is never
    /// produced by a default nobody chose.
    pub libver: LibVer,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            string_class: StringClass::Char,
            compression: Compression::None,
            invalid_name_policy: InvalidNamePolicy::Error,
            null_policy: NullPolicy::EmptyStructArray,
            unit_variant_encoding: UnitVariantEncoding::Name,
            empty_sequence_policy: EmptySequencePolicy::DoubleArray,
            unsupported_policy: UnsupportedPolicy::Error,
            one_dimensional_mode: OneDimensionalMode::ColumnVector,
            row_major_policy: RowMajorPolicy::ReorderToColumnMajor,
            empty_marker_encoding: EmptyMarkerEncoding::DataAsDims,
            libver: LibVer::V18,
        }
    }
}

impl Options {
    /// Construct options that emit the modern MATLAB `string` class via
    /// `mxOPAQUE_CLASS`, which real MATLAB's `save -v7.3` produces (and which
    /// the BEVE → MAT walker has historically used) where the default writes
    /// `char`.
    ///
    /// That is now the only difference from [`Options::default`]: the empty
    /// marker this used to override is the default encoding.
    pub fn with_modern_strings() -> Self {
        Self {
            string_class: StringClass::String,
            ..Self::default()
        }
    }
}

#[cfg(feature = "serde")]
fn serialize_deflate_level<S: Serializer>(
    level: &DeflateLevel,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_u8(level.get())
}

#[cfg(feature = "serde")]
fn deserialize_deflate_level<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<DeflateLevel, D::Error> {
    DeflateLevel::try_from(u8::deserialize(deserializer)?).map_err(serde::de::Error::custom)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "serde")]
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[cfg(feature = "serde")]
    #[case::level_zero(DeflateLevel::Zero)]
    #[case::level_six(DeflateLevel::Six)]
    #[case::level_nine(DeflateLevel::Nine)]
    fn serializes_typed_deflate_as_a_number(#[case] level: DeflateLevel) {
        let compression = Compression::Deflate {
            level,
            shuffle: true,
        };
        let value = serde_json::json!({"Deflate": {"level": level.get(), "shuffle": true}});
        assert_eq!(serde_json::to_value(compression).unwrap(), value);
        assert_eq!(
            serde_json::from_value::<Compression>(value).unwrap(),
            compression
        );
    }

    #[rstest]
    #[cfg(feature = "serde")]
    #[case::above_level_limit(10)]
    #[case::byte_limit(u8::MAX)]
    fn rejects_invalid_deflate_during_deserialization(#[case] level: u8) {
        let value = serde_json::json!({"Deflate": {"level": level, "shuffle": true}});
        let error = serde_json::from_value::<Compression>(value).unwrap_err();
        assert_eq!(error.classify(), serde_json::error::Category::Data);
        assert_eq!(
            error.to_string(),
            crate::FormatError::InvalidDeflateLevel {
                level: u32::from(level),
            }
            .to_string()
        );
    }

    /// The encoding defaults that have not moved since the serde writer was the
    /// only consumer. `null_policy` and `empty_marker_encoding` have (see the
    /// 0.30 and 0.34 changelogs), so this is no longer a statement about the
    /// whole struct.
    #[test]
    fn encoding_defaults_match_the_legacy_serde_writer() {
        let o = Options::default();
        assert_eq!(o.string_class, StringClass::Char);
        assert_eq!(o.invalid_name_policy, InvalidNamePolicy::Error);
    }

    /// The defaults that describe the *file* rather than the values in it are
    /// the ones MATLAB itself writes: an empty array is a `uint64` dataset
    /// holding its own dimensions, and the format is old enough for the HDF5
    /// 1.8.12 that MATLAB linked before R2021b.
    #[test]
    fn file_defaults_match_what_matlab_writes() {
        let o = Options::default();
        assert_eq!(o.empty_marker_encoding, EmptyMarkerEncoding::DataAsDims);
        assert_eq!(o.libver, LibVer::V18);
    }

    /// The two knobs added in 0.30 default to what the writer did before them,
    /// so only `null_policy` changes an existing caller's output.
    #[test]
    fn serde_only_knobs_default_to_previous_behavior() {
        let o = Options::default();
        assert_eq!(o.unit_variant_encoding, UnitVariantEncoding::Name);
        assert_eq!(o.empty_sequence_policy, EmptySequencePolicy::DoubleArray);
    }

    #[test]
    fn modern_strings_constructor() {
        let o = Options::with_modern_strings();
        assert_eq!(o.string_class, StringClass::String);
        assert_eq!(o.empty_marker_encoding, EmptyMarkerEncoding::DataAsDims);
    }
}
