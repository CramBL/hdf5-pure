//! MATLAB class strings used in the `MATLAB_class` attribute.
//!
//! These correspond to MATLAB's built-in numeric, character, logical, struct,
//! and cell classes. Other MATLAB classes (`string`, `function_handle`,
//! user-defined `classdef` objects, …) exist but are not emitted by the
//! serializer in this release.

use super::error::MatError;

/// A recognized `MATLAB_class` value.
///
/// Non-exhaustive: MATLAB has classes this crate does not yet recognize (see the
/// module docs), so match with a `_` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MatClass {
    /// IEEE 754 64-bit float.
    Double,
    /// IEEE 754 32-bit float.
    Single,
    /// Signed 8-bit integer.
    Int8,
    /// Signed 16-bit integer.
    Int16,
    /// Signed 32-bit integer.
    Int32,
    /// Signed 64-bit integer.
    Int64,
    /// Unsigned 8-bit integer.
    UInt8,
    /// Unsigned 16-bit integer.
    UInt16,
    /// Unsigned 32-bit integer.
    UInt32,
    /// Unsigned 64-bit integer.
    UInt64,
    /// UTF-16 character array.
    Char,
    /// Boolean / logical (stored as `uint8`).
    Logical,
    /// Struct (HDF5 group with `MATLAB_fields`).
    Struct,
    /// Heterogeneous cell array. Stored as object references resolved against
    /// a hidden `#refs#` group (one entry per cell slot).
    Cell,
}

impl MatClass {
    /// The exact string stored in the `MATLAB_class` attribute.
    pub fn as_str(self) -> &'static str {
        match self {
            MatClass::Double => "double",
            MatClass::Single => "single",
            MatClass::Int8 => "int8",
            MatClass::Int16 => "int16",
            MatClass::Int32 => "int32",
            MatClass::Int64 => "int64",
            MatClass::UInt8 => "uint8",
            MatClass::UInt16 => "uint16",
            MatClass::UInt32 => "uint32",
            MatClass::UInt64 => "uint64",
            MatClass::Char => "char",
            MatClass::Logical => "logical",
            MatClass::Struct => "struct",
            MatClass::Cell => "cell",
        }
    }

    /// Parse a `MATLAB_class` attribute value.
    pub fn parse(s: &str) -> Result<Self, MatError> {
        let trimmed = trim_null_padding(s);
        Ok(match trimmed {
            "double" => MatClass::Double,
            "single" => MatClass::Single,
            "int8" => MatClass::Int8,
            "int16" => MatClass::Int16,
            "int32" => MatClass::Int32,
            "int64" => MatClass::Int64,
            "uint8" => MatClass::UInt8,
            "uint16" => MatClass::UInt16,
            "uint32" => MatClass::UInt32,
            "uint64" => MatClass::UInt64,
            "char" => MatClass::Char,
            "logical" => MatClass::Logical,
            "struct" => MatClass::Struct,
            "cell" => MatClass::Cell,
            other => return Err(MatError::UnknownClass(other.to_string())),
        })
    }

    /// The size in bytes of a single element of this numeric class.
    /// Returns `None` for non-numeric classes (Char, Struct, Cell).
    pub fn elem_size(self) -> Option<usize> {
        Some(match self {
            MatClass::Double => 8,
            MatClass::Single => 4,
            MatClass::Int8 | MatClass::UInt8 | MatClass::Logical => 1,
            MatClass::Int16 | MatClass::UInt16 | MatClass::Char => 2,
            MatClass::Int32 | MatClass::UInt32 => 4,
            MatClass::Int64 | MatClass::UInt64 => 8,
            MatClass::Struct | MatClass::Cell => return None,
        })
    }
}

fn trim_null_padding(s: &str) -> &str {
    s.trim_end_matches('\0').trim_end()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_class_strings() {
        for c in [
            MatClass::Double,
            MatClass::Single,
            MatClass::Int8,
            MatClass::Int16,
            MatClass::Int32,
            MatClass::Int64,
            MatClass::UInt8,
            MatClass::UInt16,
            MatClass::UInt32,
            MatClass::UInt64,
            MatClass::Char,
            MatClass::Logical,
            MatClass::Struct,
            MatClass::Cell,
        ] {
            assert_eq!(MatClass::parse(c.as_str()).unwrap(), c);
        }
    }

    #[test]
    fn parse_tolerates_null_padding() {
        assert_eq!(MatClass::parse("double\0\0\0").unwrap(), MatClass::Double);
    }

    #[test]
    fn unknown_class_is_error() {
        let err = MatClass::parse("datetime").unwrap_err();
        let MatError::UnknownClass(class) = err else {
            panic!("expected UnknownClass, got {err:?}");
        };
        assert_eq!(class, "datetime");
    }
}
