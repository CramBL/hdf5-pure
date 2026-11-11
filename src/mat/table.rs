//! Public typed views over decoded MATLAB `table` and `timetable` variables.
//!
//! A MATLAB `table` decodes so that each column is addressable by its variable
//! name, which gives two ways to read one:
//!
//! 1. **Known schema** — deserialize straight into your own struct whose fields
//!    are the columns (field name = variable name):
//!
//! ```no_run
//! # #[cfg(feature = "serde")] {
//! use hdf5_pure::mat::{self, MatDatetime};
//! use serde::Deserialize;
//!
//! #[derive(Deserialize)]
//! struct Trip { Time: MatDatetime, Distance: Vec<f64>, City: Vec<String> }
//!
//! #[derive(Deserialize)]
//! struct File { trips: Trip }
//! let f: File = mat::from_bytes(&std::fs::read("trips.mat").unwrap()).unwrap();
//! # let _ = f.trips.Distance;
//! # }
//! ```
//!
//! 2. **Unknown schema** — deserialize into [`MatTable`] and inspect the columns
//!    dynamically through the [`MatColumn`] enum:
//!
//! ```no_run
//! # #[cfg(feature = "serde")] {
//! use hdf5_pure::mat::{self, MatTable, MatColumn};
//! use serde::Deserialize;
//!
//! #[derive(Deserialize)]
//! struct File { trips: MatTable }
//! let f: File = mat::from_bytes(&std::fs::read("trips.mat").unwrap()).unwrap();
//! for (name, col) in f.trips.columns() {
//!     match col {
//!         MatColumn::Numeric(m) => println!("{name}: {} numbers", m.data().len()),
//!         MatColumn::Text(t) => println!("{name}: {} strings", t.len()),
//!         _ => {}
//!     }
//! }
//! # }
//! ```
//!
//! Limitations: a table's `Properties` metadata (description, variable units and
//! descriptions, dimension names, user data) is not yet surfaced — only the
//! columns, row names, and timetable row-times are read. The typed-struct path
//! (option 1) deserializes from the columns alone; a struct used there must not
//! set `#[serde(deny_unknown_fields)]`, and a homogeneous collection target
//! (`HashMap<String, _>`) only works when every column has that element type.

use core::fmt;

use serde::Deserialize;
use serde::de::{self, Deserializer, MapAccess, Visitor};

use crate::mat::de::mcos_reader::TABLE_META_KEY;
use crate::mat::matrix::Matrix;
use crate::mat::opaque::{MatCategorical, MatDatetime, MatDuration};

/// Deserializer name that asks the MAT value deserializer to present the
/// underlying column value as a `{ kind, value }` pair so the right
/// [`MatColumn`] variant can be built with full type fidelity.
pub(crate) const MATCOLUMN_SENTINEL: &str = "__hdf5_pure_mat_MatColumn__";

/// Deserializer name that asks for a table/timetable's full field map
/// *including* the reserved metadata entry. Every other deserialization target
/// (a user row-struct, a map) has that entry filtered out, so only `MatTable` /
/// `MatTimetable` observe it.
pub(crate) const MATTABLE_SENTINEL: &str = "__hdf5_pure_mat_MatTable__";

/// One column of a decoded MATLAB [`MatTable`] / [`MatTimetable`].
///
/// Numeric columns of every MATLAB class (`double`, `single`, integers,
/// `logical`) are surfaced through [`MatColumn::Numeric`] as `f64`; for exact
/// integer width read the column through the typed-struct path instead
/// (`column: Vec<i32>`). Columns whose element type this enum does not model —
/// `cell`, `struct`, user objects, nested tables — surface as
/// [`MatColumn::Other`] and are likewise reachable through the typed-struct
/// path.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MatColumn {
    /// A real numeric (or `logical`-as-number) column, coerced to `f64`. The
    /// backing [`Matrix`] keeps the column's shape; a plain column is `N×1`, a
    /// matrix-valued variable `N×k`.
    Numeric(Matrix<f64>),
    /// A `logical` column.
    Logical(Vec<bool>),
    /// A `string` or `char` text column.
    Text(Vec<String>),
    /// A `datetime` column.
    Datetime(MatDatetime),
    /// A `duration` column.
    Duration(MatDuration),
    /// A `categorical` column.
    Categorical(MatCategorical),
    /// A column whose element type this enum does not model (`cell`, `struct`,
    /// a user class, a nested table). Read it through the typed-struct path.
    Other,
}

impl MatColumn {
    /// A short tag for the column kind (`"numeric"`, `"text"`, …).
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            MatColumn::Numeric(_) => "numeric",
            MatColumn::Logical(_) => "logical",
            MatColumn::Text(_) => "text",
            MatColumn::Datetime(_) => "datetime",
            MatColumn::Duration(_) => "duration",
            MatColumn::Categorical(_) => "categorical",
            MatColumn::Other => "other",
        }
    }

    /// The numeric values (row-major) if this is a [`MatColumn::Numeric`].
    #[must_use]
    pub fn as_f64(&self) -> Option<&[f64]> {
        match self {
            MatColumn::Numeric(m) => Some(m.data()),
            _ => None,
        }
    }

    /// The backing matrix if this is a [`MatColumn::Numeric`] (use for a
    /// matrix-valued `N×k` column).
    #[must_use]
    pub fn as_matrix(&self) -> Option<&Matrix<f64>> {
        match self {
            MatColumn::Numeric(m) => Some(m),
            _ => None,
        }
    }

    /// The boolean values if this is a [`MatColumn::Logical`].
    #[must_use]
    pub fn as_bool(&self) -> Option<&[bool]> {
        match self {
            MatColumn::Logical(v) => Some(v),
            _ => None,
        }
    }

    /// The strings if this is a [`MatColumn::Text`].
    #[must_use]
    pub fn as_strings(&self) -> Option<&[String]> {
        match self {
            MatColumn::Text(v) => Some(v),
            _ => None,
        }
    }

    /// The decoded value if this is a [`MatColumn::Datetime`].
    #[must_use]
    pub fn as_datetime(&self) -> Option<&MatDatetime> {
        match self {
            MatColumn::Datetime(d) => Some(d),
            _ => None,
        }
    }

    /// The decoded value if this is a [`MatColumn::Duration`].
    #[must_use]
    pub fn as_duration(&self) -> Option<&MatDuration> {
        match self {
            MatColumn::Duration(d) => Some(d),
            _ => None,
        }
    }

    /// The decoded value if this is a [`MatColumn::Categorical`].
    #[must_use]
    pub fn as_categorical(&self) -> Option<&MatCategorical> {
        match self {
            MatColumn::Categorical(c) => Some(c),
            _ => None,
        }
    }
}

impl<'de> Deserialize<'de> for MatColumn {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // The MAT deserializer recognizes this sentinel and presents the column
        // as a `{ kind, value }` pair (see `value_de`); any other deserializer
        // would just see a struct with those two fields, which is harmless.
        deserializer.deserialize_struct(MATCOLUMN_SENTINEL, &["kind", "value"], MatColumnVisitor)
    }
}

struct MatColumnVisitor;

impl<'de> Visitor<'de> for MatColumnVisitor {
    type Value = MatColumn;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a MATLAB table column")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<MatColumn, A::Error> {
        // First entry is the kind tag; the value's type then depends on it.
        let kind_key: Option<String> = map.next_key()?;
        if kind_key.as_deref() != Some("kind") {
            return Err(de::Error::custom("MatColumn: expected `kind` first"));
        }
        let kind: String = map.next_value()?;

        let value_key: Option<String> = map.next_key()?;
        if value_key.as_deref() != Some("value") {
            return Err(de::Error::custom(
                "MatColumn: expected `value` after `kind`",
            ));
        }
        let column = match kind.as_str() {
            "numeric" => MatColumn::Numeric(map.next_value()?),
            "logical" => MatColumn::Logical(map.next_value()?),
            "text" => MatColumn::Text(map.next_value()?),
            "datetime" => MatColumn::Datetime(map.next_value()?),
            "duration" => MatColumn::Duration(map.next_value()?),
            "categorical" => MatColumn::Categorical(map.next_value()?),
            _ => {
                let _: de::IgnoredAny = map.next_value()?;
                MatColumn::Other
            }
        };
        // Drain any remaining keys to satisfy the MapAccess contract.
        while map.next_key::<de::IgnoredAny>()?.is_some() {
            let _: de::IgnoredAny = map.next_value()?;
        }
        Ok(column)
    }
}

// ---------------------------------------------------------------------------
// MatTable
// ---------------------------------------------------------------------------

/// A decoded MATLAB `table`: named columns plus optional row names.
///
/// Deserialize a table variable into this type to inspect it without knowing
/// its schema up front; columns keep their declaration order.
#[derive(Debug, Clone, PartialEq)]
pub struct MatTable {
    columns: Vec<(String, MatColumn)>,
    num_rows: usize,
    row_names: Vec<String>,
}

impl MatTable {
    /// The number of rows.
    #[must_use]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// The number of variables (columns).
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.columns.len()
    }

    /// The variable (column) names in declaration order.
    pub fn variable_names(&self) -> impl Iterator<Item = &str> {
        self.columns.iter().map(|(n, _)| n.as_str())
    }

    /// The row names, or an empty slice when the table has none.
    #[must_use]
    pub fn row_names(&self) -> &[String] {
        &self.row_names
    }

    /// The columns as `(name, column)` pairs in declaration order.
    pub fn columns(&self) -> impl Iterator<Item = (&str, &MatColumn)> {
        self.columns.iter().map(|(n, c)| (n.as_str(), c))
    }

    /// The column with the given variable name.
    #[must_use]
    pub fn column(&self, name: &str) -> Option<&MatColumn> {
        self.columns.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }
}

impl<'de> Deserialize<'de> for MatTable {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // The sentinel requests the field map *with* the reserved metadata entry
        // (a plain struct/map target never sees it).
        let parts = deserializer.deserialize_struct(MATTABLE_SENTINEL, &[], TabularVisitor)?;
        Ok(MatTable {
            columns: parts.columns,
            num_rows: parts.num_rows,
            row_names: parts.meta.row_names,
        })
    }
}

// ---------------------------------------------------------------------------
// MatTimetable
// ---------------------------------------------------------------------------

/// A decoded MATLAB `timetable`: named columns plus a row-time vector.
///
/// The row times are a [`MatColumn::Datetime`] or [`MatColumn::Duration`]. A
/// timetable built from a sample rate or time step stores only its start and
/// step, and the row times are computed from them as MATLAB does. A calendar
/// step such as `calmonths(1)` has no fixed length and surfaces as
/// [`MatColumn::Other`].
#[derive(Debug, Clone, PartialEq)]
pub struct MatTimetable {
    columns: Vec<(String, MatColumn)>,
    num_rows: usize,
    row_times: Option<MatColumn>,
}

impl MatTimetable {
    /// The number of rows.
    #[must_use]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// The number of variables (columns).
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.columns.len()
    }

    /// The variable (column) names in declaration order.
    pub fn variable_names(&self) -> impl Iterator<Item = &str> {
        self.columns.iter().map(|(n, _)| n.as_str())
    }

    /// The row-time column, if present.
    #[must_use]
    pub fn row_times(&self) -> Option<&MatColumn> {
        self.row_times.as_ref()
    }

    /// The columns as `(name, column)` pairs in declaration order.
    pub fn columns(&self) -> impl Iterator<Item = (&str, &MatColumn)> {
        self.columns.iter().map(|(n, c)| (n.as_str(), c))
    }

    /// The column with the given variable name.
    #[must_use]
    pub fn column(&self, name: &str) -> Option<&MatColumn> {
        self.columns.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }
}

impl<'de> Deserialize<'de> for MatTimetable {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let parts = deserializer.deserialize_struct(MATTABLE_SENTINEL, &[], TabularVisitor)?;
        Ok(MatTimetable {
            columns: parts.columns,
            num_rows: parts.num_rows,
            row_times: parts.meta.row_times,
        })
    }
}

// ---------------------------------------------------------------------------
// Shared tabular deserialization
// ---------------------------------------------------------------------------

/// Metadata carried under [`TABLE_META_KEY`].
#[derive(Default, Deserialize)]
struct TabularMeta {
    #[serde(default)]
    row_names: Vec<String>,
    #[serde(default)]
    row_times: Option<MatColumn>,
    #[serde(default)]
    num_rows: Option<f64>,
}

struct TabularParts {
    columns: Vec<(String, MatColumn)>,
    num_rows: usize,
    meta: TabularMeta,
}

struct TabularVisitor;

impl<'de> Visitor<'de> for TabularVisitor {
    type Value = TabularParts;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a MATLAB table or timetable")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<TabularParts, A::Error> {
        let mut columns: Vec<(String, MatColumn)> = Vec::new();
        let mut meta = TabularMeta::default();
        while let Some(key) = map.next_key::<String>()? {
            if key == TABLE_META_KEY {
                meta = map.next_value()?;
            } else {
                columns.push((key, map.next_value()?));
            }
        }
        // Prefer MATLAB's stored row count; fall back to the first column's
        // length so a table with no `nrows` property still reports its height.
        let num_rows = match meta.num_rows {
            Some(n) if n >= 0.0 => rows_from_f64(n),
            _ => columns.first().map_or(0, |(_, c)| column_len(c)),
        };
        Ok(TabularParts {
            columns,
            num_rows,
            meta,
        })
    }
}

/// Convert MATLAB's stored row count (a non-negative `double`) to `usize`.
/// Row counts are small whole numbers, so the truncating cast is exact.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a table's row count is a small non-negative integer stored as f64; \
              the caller guards n >= 0.0 and the value is well within usize"
)]
fn rows_from_f64(n: f64) -> usize {
    n as usize
}

/// The row count a column implies (used only when no explicit count is stored).
fn column_len(column: &MatColumn) -> usize {
    match column {
        // A plain column is `N×1`; a `1×N` row-vector (a column read from a
        // true 1-D dataset) has its length in `cols`. Take the longer extent.
        MatColumn::Numeric(m) => m.rows().max(m.cols()),
        MatColumn::Logical(v) => v.len(),
        MatColumn::Text(v) => v.len(),
        MatColumn::Datetime(d) => d.millis_utc.len(),
        MatColumn::Duration(d) => d.millis.len(),
        MatColumn::Categorical(c) => c.codes.len(),
        MatColumn::Other => 0,
    }
}
