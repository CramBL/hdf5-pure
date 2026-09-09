//! Top-level serializer that enforces "the root value must be a struct".
//!
//! Each top-level field becomes a MATLAB variable in the produced `.mat`
//! file. A flat `HashMap<String, T>` is also accepted, matching
//! `scipy.io.savemat`'s dict-at-root convention.

use serde::ser::{Impossible, Serialize, SerializeMap, SerializeStruct, Serializer};

use crate::mat::error::MatError;
use crate::mat::options::Options;

use super::emit::{emit_file, emit_file_to};
use super::emit_with_builder::{emit_file_with_options, emit_file_with_options_to};
use super::value_ser::{ValueSerializer, to_value};
use crate::mat::value::MatValue;

/// Serialize `value` to MAT v7.3 file bytes (with 512-byte userblock).
pub fn to_bytes<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, MatError> {
    let options = Options::default();
    let fields = value.serialize(RootSerializer::new(&options))?;
    emit_file(fields)
}

/// Like [`to_bytes`] but with explicit options. Use this to opt into the
/// modern `string` class, name sanitization, deflate compression, etc.
pub fn to_bytes_with_options<T: Serialize + ?Sized>(
    value: &T,
    options: &Options,
) -> Result<Vec<u8>, MatError> {
    let fields = value.serialize(RootSerializer::new(options))?;
    emit_file_with_options(fields, options)
}

/// Serialize `value` straight onto `w`, without assembling the file in memory.
pub fn to_writer<T: Serialize + ?Sized, W: std::io::Write>(
    value: &T,
    w: W,
) -> Result<(), MatError> {
    let options = Options::default();
    let fields = value.serialize(RootSerializer::new(&options))?;
    emit_file_to(fields, w)
}

/// Like [`to_writer`] but with explicit options.
pub fn to_writer_with_options<T: Serialize + ?Sized, W: std::io::Write>(
    value: &T,
    options: &Options,
    w: W,
) -> Result<(), MatError> {
    let fields = value.serialize(RootSerializer::new(options))?;
    emit_file_with_options_to(fields, options, w)
}

/// Serialize `value` to `path`, streaming the file rather than buffering it.
///
/// The value is lowered to its field tree *before* the destination is created,
/// so a value this crate refuses — a non-string map key, an unsupported type, an
/// invalid name under [`InvalidNamePolicy::Error`](crate::mat::InvalidNamePolicy)
/// — leaves an existing file at `path` untouched. Only a failure during emission
/// can leave a partial file, which is inherent to writing without buffering.
pub fn to_path<T: Serialize + ?Sized, P: AsRef<std::path::Path>>(
    value: &T,
    path: P,
) -> Result<(), MatError> {
    let options = Options::default();
    let fields = value.serialize(RootSerializer::new(&options))?;
    emit_file_to(fields, create(path)?)
}

/// Like [`to_path`] but with explicit options.
pub fn to_path_with_options<T: Serialize + ?Sized, P: AsRef<std::path::Path>>(
    value: &T,
    path: P,
    options: &Options,
) -> Result<(), MatError> {
    let fields = value.serialize(RootSerializer::new(options))?;
    emit_file_with_options_to(fields, options, create(path)?)
}

/// Create the destination. Called only once the value is known to be
/// serializable, since creating it truncates whatever was there.
fn create<P: AsRef<std::path::Path>>(path: P) -> Result<std::fs::File, MatError> {
    std::fs::File::create(path).map_err(MatError::Io)
}

/// The root serializer. Produces `Vec<(field_name, MatValue)>`.
pub(crate) struct RootSerializer<'a> {
    opts: &'a Options,
}

impl<'a> RootSerializer<'a> {
    pub(crate) fn new(opts: &'a Options) -> Self {
        Self { opts }
    }

    /// A `None` / `()` / unit struct at the root.
    ///
    /// The root names no slot, so it is the variable namespace rather than a
    /// value in one. A null namespace is an empty namespace: both
    /// [`NullPolicy::Omit`] and [`NullPolicy::EmptyStructArray`] write a valid
    /// file with no variables, which is byte-identical to what an empty root map
    /// or a fieldless struct writes. `EmptyStructArray` cannot do otherwise,
    /// since `struct([])` needs a variable name to hang on.
    ///
    /// [`NullPolicy::Error`] is the one policy that *is* expressible here, and it
    /// routes through the same lowering as every other slot so it refuses with the
    /// same message. Skipping that was the policy failing at its only purpose.
    ///
    /// [`NullPolicy::Omit`]: crate::mat::NullPolicy::Omit
    /// [`NullPolicy::EmptyStructArray`]: crate::mat::NullPolicy::EmptyStructArray
    /// [`NullPolicy::Error`]: crate::mat::NullPolicy::Error
    fn root_null(self) -> Result<Vec<(String, MatValue)>, MatError> {
        super::value_ser::null_value(self.opts)?;
        Ok(Vec::new())
    }
}

impl<'a> Serializer for RootSerializer<'a> {
    type Ok = Vec<(String, MatValue)>;
    type Error = MatError;

    type SerializeSeq = Impossible<Vec<(String, MatValue)>, MatError>;
    type SerializeTuple = Impossible<Vec<(String, MatValue)>, MatError>;
    type SerializeTupleStruct = Impossible<Vec<(String, MatValue)>, MatError>;
    type SerializeTupleVariant = Impossible<Vec<(String, MatValue)>, MatError>;
    type SerializeMap = RootMapSer<'a>;
    type SerializeStruct = RootStructSer<'a>;
    type SerializeStructVariant = Impossible<Vec<(String, MatValue)>, MatError>;

    fn serialize_bool(self, _: bool) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_i8(self, _: i8) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_i16(self, _: i16) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_i32(self, _: i32) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_i64(self, _: i64) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_i128(self, _: i128) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_u8(self, _: u8) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_u16(self, _: u16) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_u32(self, _: u32) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_u64(self, _: u64) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_u128(self, _: u128) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_f32(self, _: f32) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_f64(self, _: f64) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_char(self, _: char) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_str(self, _: &str) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_bytes(self, _: &[u8]) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }

    fn serialize_none(self) -> Result<Self::Ok, MatError> {
        self.root_null()
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<Self::Ok, MatError> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<Self::Ok, MatError> {
        self.root_null()
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, MatError> {
        self.root_null()
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
    ) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, MatError> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<Self::Ok, MatError> {
        Err(MatError::RootMustBeStruct)
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, MatError> {
        Err(MatError::RootMustBeStruct)
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, MatError> {
        Err(MatError::RootMustBeStruct)
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<RootMapSer<'a>, MatError> {
        Ok(RootMapSer::new(self.opts))
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<RootStructSer<'a>, MatError> {
        Ok(RootStructSer::new(self.opts))
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, MatError> {
        Err(MatError::RootMustBeStruct)
    }
}

// ---------------------------------------------------------------------------
// RootStructSer — collects top-level named fields
// ---------------------------------------------------------------------------

pub(crate) struct RootStructSer<'a> {
    fields: Vec<(String, MatValue)>,
    opts: &'a Options,
}

impl<'a> RootStructSer<'a> {
    fn new(opts: &'a Options) -> Self {
        Self {
            fields: Vec::new(),
            opts,
        }
    }
}

impl SerializeStruct for RootStructSer<'_> {
    type Ok = Vec<(String, MatValue)>;
    type Error = MatError;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), MatError> {
        let v = to_value(value, self.opts)?;
        if !matches!(v, MatValue::Omit) {
            self.fields.push((key.to_owned(), v));
        }
        Ok(())
    }

    fn end(self) -> Result<Vec<(String, MatValue)>, MatError> {
        Ok(self.fields)
    }
}

// ---------------------------------------------------------------------------
// RootMapSer — accepts HashMap<String, T> at the root
// ---------------------------------------------------------------------------

pub(crate) struct RootMapSer<'a> {
    fields: Vec<(String, MatValue)>,
    pending_key: Option<String>,
    opts: &'a Options,
}

impl<'a> RootMapSer<'a> {
    fn new(opts: &'a Options) -> Self {
        Self {
            fields: Vec::new(),
            pending_key: None,
            opts,
        }
    }
}

impl SerializeMap for RootMapSer<'_> {
    type Ok = Vec<(String, MatValue)>;
    type Error = MatError;

    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), MatError> {
        let key_val = key.serialize(ValueSerializer::new(self.opts))?;
        match key_val {
            MatValue::String(s) => {
                self.pending_key = Some(s);
                Ok(())
            }
            other => Err(MatError::Custom(format!(
                "root map keys must be strings, got {}",
                other.kind()
            ))),
        }
    }

    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), MatError> {
        let k = self
            .pending_key
            .take()
            .ok_or_else(|| MatError::Custom("serialize_value before serialize_key".into()))?;
        let v = to_value(value, self.opts)?;
        if !matches!(v, MatValue::Omit) {
            self.fields.push((k, v));
        }
        Ok(())
    }

    fn end(self) -> Result<Vec<(String, MatValue)>, MatError> {
        Ok(self.fields)
    }
}
