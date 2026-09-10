//! Builder types for HDF5 datatypes, attributes, datasets, and groups.
//!
//! Extracted from `file_writer.rs` to keep modules under the line limit.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, string::ToString, vec, vec::Vec};

use core::fmt;

use core::num::{NonZeroU32, NonZeroUsize};

use crate::attribute::AttributeMessage;
use crate::chunked_write::{ChunkMeta, ChunkOptions, ChunkProvider, FilterKind, StorageAllocation};
use crate::compound::CompoundType;
use crate::convert::TryToUsize;
use crate::dataspace::{Dataspace, DataspaceType};
use crate::datatype::{
    CharacterSet, CompoundMember, Datatype, DatatypeByteOrder, EnumMember, StringPadding,
};
use crate::display::write_elided;
use crate::error::FormatError;
use crate::scaleoffset::{FillAvailability, ScaleOffset};
use crate::shared_message::DatatypeLocation;

// ---- Datatype constructors ----

pub fn make_f64_type() -> Datatype {
    Datatype::FloatingPoint {
        size: 8,
        byte_order: DatatypeByteOrder::LittleEndian,
        bit_offset: 0,
        bit_precision: 64,
        exponent_location: 52,
        exponent_size: 11,
        mantissa_location: 0,
        mantissa_size: 52,
        exponent_bias: 1023,
    }
}

pub fn make_f32_type() -> Datatype {
    Datatype::FloatingPoint {
        size: 4,
        byte_order: DatatypeByteOrder::LittleEndian,
        bit_offset: 0,
        bit_precision: 32,
        exponent_location: 23,
        exponent_size: 8,
        mantissa_location: 0,
        mantissa_size: 23,
        exponent_bias: 127,
    }
}

pub fn make_i32_type() -> Datatype {
    Datatype::FixedPoint {
        size: 4,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: true,
        bit_offset: 0,
        bit_precision: 32,
    }
}

pub fn make_i64_type() -> Datatype {
    Datatype::FixedPoint {
        size: 8,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: true,
        bit_offset: 0,
        bit_precision: 64,
    }
}

pub fn make_u8_type() -> Datatype {
    Datatype::FixedPoint {
        size: 1,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: false,
        bit_offset: 0,
        bit_precision: 8,
    }
}

pub fn make_i8_type() -> Datatype {
    Datatype::FixedPoint {
        size: 1,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: true,
        bit_offset: 0,
        bit_precision: 8,
    }
}

pub fn make_i16_type() -> Datatype {
    Datatype::FixedPoint {
        size: 2,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: true,
        bit_offset: 0,
        bit_precision: 16,
    }
}

pub fn make_u16_type() -> Datatype {
    Datatype::FixedPoint {
        size: 2,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: false,
        bit_offset: 0,
        bit_precision: 16,
    }
}

pub fn make_u32_type() -> Datatype {
    Datatype::FixedPoint {
        size: 4,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: false,
        bit_offset: 0,
        bit_precision: 32,
    }
}

pub fn make_u64_type() -> Datatype {
    Datatype::FixedPoint {
        size: 8,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: false,
        bit_offset: 0,
        bit_precision: 64,
    }
}

pub fn make_object_reference_type() -> Datatype {
    Datatype::Reference {
        size: 8,
        ref_type: crate::datatype::ReferenceType::Object,
    }
}

/// A variable-length string datatype with the given character set and
/// null-terminated padding.
///
/// The character set and padding live in the variable-length datatype's own
/// bitfields; the base element type is an 8-bit unsigned integer
/// (`H5T_STD_U8LE`), exactly the shape the reference C library and h5py emit for
/// a VL string (`H5Tvlen_create(H5T_C_S1)` stores the base as a 1-byte
/// integer). Matching it byte-for-byte is what lets the C library read these
/// datasets back into `VarLenUnicode`/`VarLenAscii` without a conversion-path
/// error.
pub fn make_vlen_string_type(charset: CharacterSet) -> Datatype {
    Datatype::VariableLength {
        is_string: true,
        padding: Some(StringPadding::NullTerminate),
        charset: Some(charset),
        base_type: Box::new(make_u8_type()),
    }
}

// ---- Compound / Enum type builders ----

/// Builder for constructing HDF5 compound (struct) datatypes.
///
/// Fields are laid out contiguously in insertion order, and the builder computes their offsets.
/// [`new`](Self::new) starts an empty builder, [`field`](Self::field) adds an arbitrary field, and
/// the typed helpers [`f64_field`](Self::f64_field), [`f32_field`](Self::f32_field),
/// [`i8_field`](Self::i8_field), [`i16_field`](Self::i16_field), [`i32_field`](Self::i32_field),
/// [`i64_field`](Self::i64_field), [`u8_field`](Self::u8_field), [`u16_field`](Self::u16_field),
/// [`u32_field`](Self::u32_field) and [`u64_field`](Self::u64_field) add a named scalar field.
/// [`build`](Self::build) rejects a builder with no fields and one whose fields pack to zero bytes.
///
/// For an `H5Tinsert`-style layout with explicit offsets and a fixed total size,
/// [`with_size`](Self::with_size) returns an [`ExplicitCompoundTypeBuilder`] instead.
///
/// # Examples
///
/// ```rust
/// use hdf5_pure::CompoundTypeBuilder;
///
/// let dt = CompoundTypeBuilder::new()
///     .i32_field("id")
///     .f64_field("value")
///     .build()?;
/// # assert_eq!(dt.type_size(), 12);
/// # Ok::<(), hdf5_pure::FormatError>(())
/// ```
pub struct CompoundTypeBuilder {
    fields: Vec<(String, Datatype)>,
}

impl CompoundTypeBuilder {
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    /// Add a named field with the given datatype.
    pub fn field(mut self, name: &str, datatype: Datatype) -> Self {
        self.fields.push((name.to_string(), datatype));
        self
    }

    /// Add an f64 field.
    pub fn f64_field(self, name: &str) -> Self {
        self.field(name, make_f64_type())
    }
    /// Add an f32 field.
    pub fn f32_field(self, name: &str) -> Self {
        self.field(name, make_f32_type())
    }
    /// Add an i32 field.
    pub fn i32_field(self, name: &str) -> Self {
        self.field(name, make_i32_type())
    }
    /// Add an i64 field.
    pub fn i64_field(self, name: &str) -> Self {
        self.field(name, make_i64_type())
    }
    /// Add a u8 field.
    pub fn u8_field(self, name: &str) -> Self {
        self.field(name, make_u8_type())
    }
    /// Add an i8 field.
    pub fn i8_field(self, name: &str) -> Self {
        self.field(name, make_i8_type())
    }
    /// Add an i16 field.
    pub fn i16_field(self, name: &str) -> Self {
        self.field(name, make_i16_type())
    }
    /// Add a u16 field.
    pub fn u16_field(self, name: &str) -> Self {
        self.field(name, make_u16_type())
    }
    /// Add a u32 field.
    pub fn u32_field(self, name: &str) -> Self {
        self.field(name, make_u32_type())
    }
    /// Add a u64 field.
    pub fn u64_field(self, name: &str) -> Self {
        self.field(name, make_u64_type())
    }

    /// Build the compound datatype, packing the fields in the order they were
    /// added.
    ///
    /// Fails with [`FormatError::EmptyCompoundType`] over no fields, and with
    /// [`FormatError::InvalidCompoundSize`] when the fields pack to zero bytes —
    /// the two things `H5Tcreate(H5T_COMPOUND, ..)` also refuses, and the two
    /// that make a datatype nothing downstream can use: every writer and reader
    /// divides raw bytes by the element size to recover an element count.
    pub fn build(self) -> Result<Datatype, FormatError> {
        if self.fields.is_empty() {
            return Err(FormatError::EmptyCompoundType);
        }
        let mut offset = 0u64;
        let mut members = Vec::with_capacity(self.fields.len());
        for (name, dt) in self.fields {
            let sz = dt.type_size();
            members.push(CompoundMember {
                name,
                byte_offset: offset,
                datatype: dt,
            });
            offset += sz as u64;
        }
        if offset == 0 {
            return Err(FormatError::InvalidCompoundSize);
        }
        Ok(Datatype::Compound {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "accumulated compound size is stored in the 4-byte datatype size field"
            )]
            size: offset as u32,
            members,
        })
    }
}

impl Default for CompoundTypeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

mod complex_component {
    pub trait Sealed {}
}

/// A scalar type that can be the component of a complex `{real, imag}` dataset.
///
/// A complex array is a two-field compound of one numeric class, so the only
/// things the layout depends on are the component's datatype and its
/// little-endian encoding — which is exactly what this trait supplies, and why
/// [`DatasetBuilder::with_complex_data`] needs nothing per width beyond a new
/// impl here.
///
/// Sealed: the impls below cover every class the crate can store, and a
/// component type outside that set could only produce a malformed file.
pub(crate) trait ComplexComponent: complex_component::Sealed + Copy {
    /// The datatype of one component, identical to what the type's
    /// `with_*_data` writer emits for a real dataset of the same class.
    fn datatype() -> Datatype;

    /// Write `self` to `dst`, exactly `size_of::<Self>()` bytes, little-endian.
    ///
    /// A pre-sized buffer rather than a `Vec` to push onto: this runs twice per
    /// complex element, and the bounds check dominated the write — about 5x on
    /// a large array.
    fn encode_le_into(self, dst: &mut [u8]);

    /// Decode one component from exactly `size_of::<Self>()` little-endian
    /// bytes. Callers slice the element out of the raw buffer first, so a
    /// wrong length is a bug here rather than bad input.
    ///
    /// Gated to match its only caller: the MAT reader is `serde`-only, while
    /// the writer half of this trait compiles unconditionally.
    #[cfg(feature = "serde")]
    fn decode_le(bytes: &[u8]) -> Self;
}

macro_rules! impl_complex_component {
    ($($ty:ty => $make:ident),* $(,)?) => {
        $(
            impl complex_component::Sealed for $ty {}
            impl ComplexComponent for $ty {
                fn datatype() -> Datatype {
                    $make()
                }
                fn encode_le_into(self, dst: &mut [u8]) {
                    dst.copy_from_slice(&self.to_le_bytes());
                }
                #[cfg(feature = "serde")]
                fn decode_le(bytes: &[u8]) -> Self {
                    Self::from_le_bytes(
                        bytes
                            .try_into()
                            .expect("caller slices exactly one component"),
                    )
                }
            }
        )*
    };
}

impl_complex_component! {
    f64 => make_f64_type,
    f32 => make_f32_type,
    i64 => make_i64_type,
    i32 => make_i32_type,
    i16 => make_i16_type,
    i8 => make_i8_type,
    u64 => make_u64_type,
    u32 => make_u32_type,
    u16 => make_u16_type,
    u8 => make_u8_type,
}

/// Builder for an HDF5 compound datatype with explicit field offsets and size.
///
/// This is the pure-Rust equivalent of creating an `H5T_COMPOUND` type and
/// inserting fields with `H5Tinsert`. [`build`](Self::build) validates field
/// names, bounds, and overlap before returning a datatype.
///
/// Each [`field(name, byte_offset, datatype)`](Self::field) (and the typed
/// [`f64_field(name, byte_offset)`](Self::f64_field) and its siblings) places a field at a chosen
/// offset within the fixed total size given to [`CompoundTypeBuilder::with_size`].
///
/// # Examples
///
/// ```rust
/// use hdf5_pure::CompoundTypeBuilder;
///
/// let dt = CompoundTypeBuilder::with_size(16)
///     .i32_field("id", 0)
///     .f64_field("value", 8)
///     .build()?;
/// # assert_eq!(dt.type_size(), 16);
/// # Ok::<(), hdf5_pure::FormatError>(())
/// ```
pub struct ExplicitCompoundTypeBuilder {
    size: u32,
    fields: Vec<CompoundMember>,
}

impl ExplicitCompoundTypeBuilder {
    /// Add a field at an explicit byte offset.
    pub fn field(mut self, name: &str, byte_offset: u64, datatype: Datatype) -> Self {
        self.fields.push(CompoundMember {
            name: name.to_string(),
            byte_offset,
            datatype,
        });
        self
    }

    /// Add an f64 field at an explicit byte offset.
    pub fn f64_field(self, name: &str, byte_offset: u64) -> Self {
        self.field(name, byte_offset, make_f64_type())
    }

    /// Add an f32 field at an explicit byte offset.
    pub fn f32_field(self, name: &str, byte_offset: u64) -> Self {
        self.field(name, byte_offset, make_f32_type())
    }

    /// Add an i32 field at an explicit byte offset.
    pub fn i32_field(self, name: &str, byte_offset: u64) -> Self {
        self.field(name, byte_offset, make_i32_type())
    }

    /// Add an i64 field at an explicit byte offset.
    pub fn i64_field(self, name: &str, byte_offset: u64) -> Self {
        self.field(name, byte_offset, make_i64_type())
    }

    /// Add a u8 field at an explicit byte offset.
    pub fn u8_field(self, name: &str, byte_offset: u64) -> Self {
        self.field(name, byte_offset, make_u8_type())
    }

    /// Add an i8 field at an explicit byte offset.
    pub fn i8_field(self, name: &str, byte_offset: u64) -> Self {
        self.field(name, byte_offset, make_i8_type())
    }

    /// Add an i16 field at an explicit byte offset.
    pub fn i16_field(self, name: &str, byte_offset: u64) -> Self {
        self.field(name, byte_offset, make_i16_type())
    }

    /// Add a u16 field at an explicit byte offset.
    pub fn u16_field(self, name: &str, byte_offset: u64) -> Self {
        self.field(name, byte_offset, make_u16_type())
    }

    /// Add a u32 field at an explicit byte offset.
    pub fn u32_field(self, name: &str, byte_offset: u64) -> Self {
        self.field(name, byte_offset, make_u32_type())
    }

    /// Add a u64 field at an explicit byte offset.
    pub fn u64_field(self, name: &str, byte_offset: u64) -> Self {
        self.field(name, byte_offset, make_u64_type())
    }

    /// Validate and build the compound datatype.
    pub fn build(mut self) -> Result<Datatype, crate::error::FormatError> {
        use crate::error::FormatError;

        if self.size == 0 {
            return Err(FormatError::InvalidCompoundSize);
        }
        if self.fields.is_empty() {
            return Err(FormatError::EmptyCompoundType);
        }

        for (index, field) in self.fields.iter().enumerate() {
            if self.fields[..index]
                .iter()
                .any(|earlier| earlier.name == field.name)
            {
                return Err(FormatError::DuplicateCompoundField(field.name.clone()));
            }
            let field_size = field.datatype.type_size();
            let end = field.byte_offset.checked_add(u64::from(field_size));
            if field_size == 0 || end.is_none_or(|end| end > u64::from(self.size)) {
                return Err(FormatError::CompoundFieldOutOfBounds {
                    name: field.name.clone(),
                    offset: field.byte_offset,
                    field_size,
                    compound_size: self.size,
                });
            }
        }

        // Stable on purpose: two fields at one offset are rejected as an overlap
        // just below, and stability is what makes that error name them in
        // declaration order rather than an arbitrary one.
        self.fields.sort_by_key(|field| field.byte_offset);
        for fields in self.fields.windows(2) {
            let first_end = fields[0].byte_offset + u64::from(fields[0].datatype.type_size());
            if first_end > fields[1].byte_offset {
                return Err(FormatError::CompoundFieldOverlap {
                    first: fields[0].name.clone(),
                    second: fields[1].name.clone(),
                });
            }
        }

        Ok(Datatype::Compound {
            size: self.size,
            members: self.fields,
        })
    }
}

impl CompoundTypeBuilder {
    /// Create a compound builder with an explicit total size and field offsets.
    pub fn with_size(size: u32) -> ExplicitCompoundTypeBuilder {
        ExplicitCompoundTypeBuilder {
            size,
            fields: Vec::new(),
        }
    }
}

/// Builder for constructing HDF5 enumeration datatypes over an integer base type.
///
/// Start with [`i32_based`](Self::i32_based) or [`u8_based`](Self::u8_based), or with
/// [`with_base`](Self::with_base) for any other integer type. Add members with
/// [`value(name, i32)`](Self::value), [`u8_value(name, u8)`](Self::u8_value),
/// [`i64_value(name, i64)`](Self::i64_value) for a value wider than an `i32`, or
/// [`raw_value(name, bytes)`](Self::raw_value) for one wider than an `i64`. Then call
/// [`build`](Self::build).
///
/// # Examples
///
/// ```rust
/// use hdf5_pure::EnumTypeBuilder;
///
/// let dt = EnumTypeBuilder::i32_based()
///     .value("Red", 0)
///     .value("Green", 1)
///     .value("Blue", 2)
///     .build()?;
/// # assert_eq!(dt.type_size(), 4);
/// # Ok::<(), hdf5_pure::FormatError>(())
/// ```
pub struct EnumTypeBuilder {
    base_type: Datatype,
    members: Vec<(String, PendingEnumValue)>,
}

/// A member value awaiting the base type's width, resolved by
/// [`EnumTypeBuilder::build`].
enum PendingEnumValue {
    /// An integer, encoded little-endian into the base type's size at build.
    Int(i64),
    /// Raw little-endian bytes, which must already match the base type's size.
    Raw(Vec<u8>),
}

impl EnumTypeBuilder {
    /// Create a new enum builder with i32 base type.
    pub fn i32_based() -> Self {
        Self::with_base(make_i32_type())
    }

    /// Create a new enum builder with u8 base type.
    pub fn u8_based() -> Self {
        Self::with_base(make_u8_type())
    }

    /// Create an enum builder over an arbitrary integer base type, such as
    /// [`make_u16_type`] or [`make_i64_type`].
    ///
    /// The enumeration's element size comes from `base_type`. A non-integer base
    /// is refused by [`build`](Self::build) with
    /// [`FormatError::EnumBaseNotInteger`].
    pub fn with_base(base_type: Datatype) -> Self {
        Self {
            base_type,
            members: Vec::new(),
        }
    }

    /// Add a named value, encoded little-endian into the base type's width.
    ///
    /// A value that does not fit the base type is refused by
    /// [`build`](Self::build) with [`FormatError::EnumMemberValueRange`].
    pub fn value(mut self, name: &str, val: i32) -> Self {
        self.members
            .push((name.to_string(), PendingEnumValue::Int(val as i64)));
        self
    }

    /// Add a named u8 value.
    pub fn u8_value(self, name: &str, val: u8) -> Self {
        self.value(name, i32::from(val))
    }

    /// Add a named value from an integer wide enough for any base type.
    pub fn i64_value(mut self, name: &str, val: i64) -> Self {
        self.members
            .push((name.to_string(), PendingEnumValue::Int(val)));
        self
    }

    /// Add a named value from its raw little-endian bytes — the form the format
    /// stores and [`EnumMember::value`] holds.
    ///
    /// `bytes` must be exactly the base type's size, or [`build`](Self::build)
    /// refuses it with [`FormatError::EnumMemberValueSize`]. Use this for a base
    /// type whose values do not fit an `i64`, or to reproduce stored bytes
    /// verbatim.
    pub fn raw_value(mut self, name: &str, bytes: &[u8]) -> Self {
        self.members
            .push((name.to_string(), PendingEnumValue::Raw(bytes.to_vec())));
        self
    }

    /// Build the enumeration datatype, resolving every member against the base
    /// type's width.
    ///
    /// Fails if the base type is not an integer, if a member's raw byte length
    /// disagrees with the base type's size, or if a member's integer value does
    /// not fit — rather than emitting a datatype message the reference C library
    /// cannot read.
    pub fn build(self) -> Result<Datatype, FormatError> {
        let size = self.base_type.type_size();
        let signed = match &self.base_type {
            Datatype::FixedPoint { signed, .. } => *signed,
            _ => return Err(FormatError::EnumBaseNotInteger),
        };
        let width = size.to_usize()?;

        let mut members = Vec::with_capacity(self.members.len());
        for (name, pending) in self.members {
            let value = match pending {
                PendingEnumValue::Raw(bytes) => {
                    if bytes.len() != width {
                        return Err(FormatError::EnumMemberValueSize(name, size, bytes.len()));
                    }
                    bytes
                }
                PendingEnumValue::Int(v) => {
                    if !int_fits(v, width, signed) {
                        return Err(FormatError::EnumMemberValueRange(name, v, size));
                    }
                    v.to_le_bytes()[..width].to_vec()
                }
            };
            members.push(EnumMember { name, value });
        }

        Ok(Datatype::Enumeration {
            size,
            base_type: Box::new(self.base_type),
            members,
        })
    }
}

/// Whether `v` is representable in `width` bytes under `signed`.
fn int_fits(v: i64, width: usize, signed: bool) -> bool {
    if width == 0 {
        return false;
    }
    if width >= 8 {
        // Every `i64` fits eight signed bytes; an unsigned eight-byte base needs
        // a non-negative value (larger magnitudes need `raw_value`).
        return signed || v >= 0;
    }
    let bits = width * 8;
    if signed {
        let min = -(1i64 << (bits - 1));
        let max = (1i64 << (bits - 1)) - 1;
        (min..=max).contains(&v)
    } else {
        let max = (1i64 << bits) - 1;
        (0..=max).contains(&v)
    }
}

// ---- Attribute helper ----

/// How a builder holds one attribute until the writer turns it into bytes.
///
/// Most callers describe an attribute by *value* and let the writer choose an
/// encoding for it — that is [`Value`](AttrSpec::Value), and the encoding it
/// gets is whatever [`build_attr_message`] picks for the variant.
///
/// [`Verbatim`](AttrSpec::Verbatim) carries an already-encoded message instead,
/// so the datatype, dataspace and element bytes reach the file exactly as
/// given. Repack uses it to copy an attribute across without routing it through
/// [`AttrValue`], which is a decoded view and cannot express a byte order, a
/// sub-width precision, a rank above one, or a string's padding — every one of
/// which the value path would therefore rewrite (see [`AttrValue`]'s docs on
/// what a decode does not recover).
///
/// The element bytes are copied as-is, so a verbatim message is only correct for
/// a datatype whose bytes mean the same thing in another file: anything holding
/// a global-heap reference or an object address must not take this path, since
/// those addresses point into the *source*. Repack decides that with
/// `attr_bytes_are_position_independent`.
pub(crate) enum AttrSpec {
    /// A decoded value the writer encodes with [`build_attr_message`].
    Value(AttrValue),
    /// An already-encoded message, written as given.
    Verbatim(AttributeMessage),
    /// A message whose datatype and dataspace are given, but whose element bytes
    /// are global-heap references the writer builds and patches from `strings`.
    ///
    /// This is the middle ground a variable-length string attribute needs. Its
    /// datatype and dataspace *can* travel — they say "variable-length UTF-8",
    /// or "scalar" — while its element bytes cannot, because they address the
    /// source file's heap. `Verbatim` would carry a dangling address across, and
    /// `Value` would re-render the datatype from whichever variant the decode
    /// chose, losing a rank above one, a padding rule other than `NULLTERM`, and
    /// the arity of a scalar in MATLAB's sequence shape. Neither alone is
    /// faithful.
    ///
    /// `message.raw_data` must already be [`vl_string_reference_bytes`] over the
    /// same `strings`, so the placeholder count matches the heap objects the
    /// writer will place.
    VerbatimVarLen {
        message: AttributeMessage,
        strings: Vec<String>,
    },
}

impl AttrSpec {
    /// The message this attribute writes, encoding a [`Value`](AttrSpec::Value)
    /// and handing back an already-encoded one unchanged.
    pub(crate) fn to_message(&self, name: &str) -> AttributeMessage {
        match self {
            Self::Value(v) => build_attr_message(name, v),
            Self::Verbatim(m) | Self::VerbatimVarLen { message: m, .. } => m.clone(),
        }
    }

    /// The variable-length strings whose global-heap collections the writer must
    /// build and patch, or `None` when the attribute needs no heap.
    pub(crate) fn var_len_strings(&self) -> Option<&[String]> {
        match self {
            Self::Value(v) => v.var_len_strings(),
            Self::VerbatimVarLen { strings, .. } => Some(strings),
            Self::Verbatim(_) => None,
        }
    }
}

/// A scalar numeric attribute: its own datatype, a scalar dataspace, and the
/// element's little-endian bytes.
fn numeric_scalar_attr(name: &str, datatype: Datatype, raw_data: &[u8]) -> AttributeMessage {
    AttributeMessage {
        name: name.to_string(),
        datatype,
        dataspace: scalar_ds(),
        raw_data: raw_data.to_vec(),
        datatype_location: DatatypeLocation::Inline,
    }
}

/// A one-dimensional numeric attribute, its elements laid out little-endian at
/// the width `datatype` declares.
///
/// `to_le_bytes` is the element's own encoder — `i16::to_le_bytes` and friends —
/// so the bytes written can only be as wide as the type they came from. Each
/// call pairs that encoder with the datatype constructor of the same width,
/// which is what makes the width a message declares and the width its bytes
/// occupy the same number.
fn numeric_array_attr<T: Copy, const N: usize>(
    name: &str,
    datatype: Datatype,
    values: &[T],
    to_le_bytes: fn(T) -> [u8; N],
) -> AttributeMessage {
    let mut raw_data = Vec::with_capacity(values.len() * N);
    for &v in values {
        raw_data.extend_from_slice(&to_le_bytes(v));
    }
    AttributeMessage {
        name: name.to_string(),
        datatype,
        dataspace: simple_1d(values.len() as u64),
        raw_data,
        datatype_location: DatatypeLocation::Inline,
    }
}

/// A fixed-width string attribute message: `values` laid out at one shared
/// width, under a `H5T_STRING { STRSIZE = width, NULLPAD, charset }` datatype
/// and `dataspace`.
///
/// `width` is the width the caller declared, or `None` to take the one the
/// values imply. Both the datatype and the element bytes read the width off
/// [`pad_fixed_strings`]'s answer rather than computing it twice, which is what
/// keeps the message describing the bytes it carries.
fn fixed_string_attr<S: AsRef<str>>(
    name: &str,
    values: &[S],
    width: Option<NonZeroU32>,
    charset: CharacterSet,
    dataspace: Dataspace,
) -> AttributeMessage {
    let (raw_data, width) = pad_fixed_strings(values, width.unwrap_or(NonZeroU32::MIN));
    AttributeMessage {
        name: name.to_string(),
        datatype: Datatype::String {
            size: width.get(),
            padding: StringPadding::NullPad,
            charset,
        },
        dataspace,
        raw_data,
        datatype_location: DatatypeLocation::Inline,
    }
}

/// One fixed-width string, under a scalar dataspace.
fn fixed_string_scalar_attr(
    name: &str,
    value: &str,
    width: Option<NonZeroU32>,
    charset: CharacterSet,
) -> AttributeMessage {
    fixed_string_attr(
        name,
        core::slice::from_ref(&value),
        width,
        charset,
        scalar_ds(),
    )
}

/// An array of fixed-width strings, under a 1-D dataspace of their own count —
/// taken here rather than passed in, so the dataspace cannot disagree with the
/// elements below it.
fn fixed_string_array_attr<S: AsRef<str>>(
    name: &str,
    values: &[S],
    width: Option<NonZeroU32>,
    charset: CharacterSet,
) -> AttributeMessage {
    let dataspace = simple_1d(values.len() as u64);
    fixed_string_attr(name, values, width, charset, dataspace)
}

pub(crate) fn build_attr_message(name: &str, value: &AttrValue) -> AttributeMessage {
    match value {
        AttrValue::F32(v) => numeric_scalar_attr(name, make_f32_type(), &v.to_le_bytes()),
        AttrValue::F32Array(a) => numeric_array_attr(name, make_f32_type(), a, f32::to_le_bytes),
        AttrValue::F64(v) => numeric_scalar_attr(name, make_f64_type(), &v.to_le_bytes()),
        AttrValue::F64Array(a) => numeric_array_attr(name, make_f64_type(), a, f64::to_le_bytes),
        AttrValue::I8(v) => numeric_scalar_attr(name, make_i8_type(), &v.to_le_bytes()),
        AttrValue::I8Array(a) => numeric_array_attr(name, make_i8_type(), a, i8::to_le_bytes),
        AttrValue::I16(v) => numeric_scalar_attr(name, make_i16_type(), &v.to_le_bytes()),
        AttrValue::I16Array(a) => numeric_array_attr(name, make_i16_type(), a, i16::to_le_bytes),
        AttrValue::I32(v) => numeric_scalar_attr(name, make_i32_type(), &v.to_le_bytes()),
        AttrValue::I32Array(a) => numeric_array_attr(name, make_i32_type(), a, i32::to_le_bytes),
        AttrValue::I64(v) => numeric_scalar_attr(name, make_i64_type(), &v.to_le_bytes()),
        AttrValue::I64Array(a) => numeric_array_attr(name, make_i64_type(), a, i64::to_le_bytes),
        AttrValue::U8(v) => numeric_scalar_attr(name, make_u8_type(), &v.to_le_bytes()),
        AttrValue::U8Array(a) => numeric_array_attr(name, make_u8_type(), a, u8::to_le_bytes),
        AttrValue::U16(v) => numeric_scalar_attr(name, make_u16_type(), &v.to_le_bytes()),
        AttrValue::U16Array(a) => numeric_array_attr(name, make_u16_type(), a, u16::to_le_bytes),
        AttrValue::U32(v) => numeric_scalar_attr(name, make_u32_type(), &v.to_le_bytes()),
        AttrValue::U32Array(a) => numeric_array_attr(name, make_u32_type(), a, u32::to_le_bytes),
        AttrValue::U64(v) => numeric_scalar_attr(name, make_u64_type(), &v.to_le_bytes()),
        AttrValue::U64Array(a) => numeric_array_attr(name, make_u64_type(), a, u64::to_le_bytes),
        AttrValue::String(s) => fixed_string_scalar_attr(name, s, None, CharacterSet::Utf8),
        AttrValue::StringSized { value, width } => {
            fixed_string_scalar_attr(name, value, Some(*width), CharacterSet::Utf8)
        }
        AttrValue::StringArray(arr) => fixed_string_array_attr(name, arr, None, CharacterSet::Utf8),
        AttrValue::StringArraySized { values, width } => {
            fixed_string_array_attr(name, values, Some(*width), CharacterSet::Utf8)
        }
        AttrValue::AsciiString(s) => fixed_string_scalar_attr(name, s, None, CharacterSet::Ascii),
        AttrValue::AsciiStringSized { value, width } => {
            fixed_string_scalar_attr(name, value, Some(*width), CharacterSet::Ascii)
        }
        AttrValue::AsciiStringArray(arr) => {
            fixed_string_array_attr(name, arr, None, CharacterSet::Ascii)
        }
        AttrValue::AsciiStringArraySized { values, width } => {
            fixed_string_array_attr(name, values, Some(*width), CharacterSet::Ascii)
        }
        AttrValue::VarLenAsciiCharArray(values) => {
            vlen_string_array_attr(name, values, make_matlab_vlen_ascii_type())
        }
        AttrValue::VarLenString(value) => vlen_string_scalar_attr(name, value, CharacterSet::Utf8),
        AttrValue::VarLenAsciiString(value) => {
            vlen_string_scalar_attr(name, value, CharacterSet::Ascii)
        }
        AttrValue::VarLenStringArray(values) => {
            vlen_string_array_attr(name, values, make_vlen_string_type(CharacterSet::Utf8))
        }
        AttrValue::VarLenAsciiStringArray(values) => {
            vlen_string_array_attr(name, values, make_vlen_string_type(CharacterSet::Ascii))
        }
    }
}

/// MATLAB v7.3 and matio expect `MATLAB_fields` and similar variable-length
/// ASCII arrays as `H5T_VLEN { H5T_STRING { STRSIZE = 1, NULLTERM, ASCII } }` —
/// a VLEN sequence of 1-byte fixed strings rather than a variable-length string.
///
/// Only the descriptor differs from [`make_vlen_string_type`]: the elements
/// underneath are [`vl_string_reference_bytes`] either way.
fn make_matlab_vlen_ascii_type() -> Datatype {
    Datatype::VariableLength {
        is_string: false,
        padding: None,
        charset: None,
        base_type: Box::new(Datatype::String {
            size: 1,
            padding: StringPadding::NullTerminate,
            charset: CharacterSet::Ascii,
        }),
    }
}

/// A variable-length string attribute: one 16-byte global-heap reference per
/// element, under the `datatype` that says how to read them back.
///
/// The datatype is a parameter because both variable-length string encodings
/// this crate writes stand over the same element bytes, and
/// [`vl_string_reference_bytes`] is the one encoder for those. The heap
/// addresses in them are placeholders the writer patches once it has placed the
/// collections.
fn vlen_string_attr(
    name: &str,
    strings: &[String],
    datatype: Datatype,
    dataspace: Dataspace,
) -> AttributeMessage {
    AttributeMessage {
        name: name.to_string(),
        raw_data: vl_string_reference_bytes(strings),
        datatype,
        dataspace,
        datatype_location: DatatypeLocation::Inline,
    }
}

/// One variable-length string under a scalar dataspace — the standard
/// `H5T_STRING` with `STRSIZE = H5T_VARIABLE`.
fn vlen_string_scalar_attr(name: &str, value: &String, charset: CharacterSet) -> AttributeMessage {
    vlen_string_attr(
        name,
        core::slice::from_ref(value),
        make_vlen_string_type(charset),
        scalar_ds(),
    )
}

/// An array of variable-length strings under `datatype`, with a 1-D dataspace of
/// their own count — taken here rather than passed in, so the dataspace cannot
/// disagree with the references below it. Both encodings go through this, which
/// is what keeps that guarantee true of MATLAB's shape as well.
fn vlen_string_array_attr(name: &str, values: &[String], datatype: Datatype) -> AttributeMessage {
    vlen_string_attr(name, values, datatype, simple_1d(values.len() as u64))
}

/// The element bytes of a variable-length string value: one 16-byte global-heap
/// reference per string, with the collection address left as a placeholder for
/// the writer to patch once it has placed the collections.
///
/// Both variable-length string encodings this crate handles share these bytes —
/// a true `H5T_STRING` with `STRSIZE = VAR`, and the `H5T_VLEN` of 1-byte
/// strings that MATLAB and matio emit — so only the datatype descriptor around
/// them differs. That is what lets repack keep a source attribute's own datatype
/// while rebuilding its payload against the destination's heap.
///
/// The object index is 1-based *within each collection*, and the split into
/// collections must match [`build_global_heap_collections`] exactly: the two
/// walk the same strings in the same order, and a divergence would point an
/// element at the wrong heap object. Keeping one encoder for every caller is
/// what holds that invariant to a single place.
pub(crate) fn vl_string_reference_bytes(strings: &[String]) -> Vec<u8> {
    let mut raw = Vec::with_capacity(strings.len() * VL_REF_SIZE);
    for (i, s) in strings.iter().enumerate() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "VLEN string length is written into the 4-byte length prefix of the variable-length reference"
        )]
        raw.extend_from_slice(&(s.len() as u32).to_le_bytes());
        raw.extend_from_slice(&0u64.to_le_bytes()); // patched later
        #[expect(
            clippy::cast_possible_truncation,
            reason = "1-based heap object index is written into the 4-byte object-index field of the variable-length reference"
        )]
        raw.extend_from_slice(&((i % MAX_HEAP_OBJECTS + 1) as u32).to_le_bytes());
    }
    raw
}

/// Maximum number of objects one global heap collection can index.
///
/// The heap-object index field is a `u16` with 0 reserved for the free-space
/// marker, so a single collection addresses at most `u16::MAX` objects. Data
/// with more objects than this is split across consecutive collections, whose
/// indices restart at 1 — the same thing the reference C library does when a
/// collection fills.
pub(crate) const MAX_HEAP_OBJECTS: usize = u16::MAX as usize;

/// Build the global heap collections holding the given strings, splitting them
/// across as many collections as their count needs.
pub(crate) fn build_global_heap_collections(strings: &[String]) -> Vec<Vec<u8>> {
    let objects: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
    build_global_heap_collections_from_bytes(&objects)
}

/// Build the global heap collections holding the given raw byte objects (no
/// UTF-8 requirement), in order. Mirrors [`build_global_heap_collections`] but
/// accepts arbitrary bytes so a faithful rewrite can carry embedded-NUL or
/// non-UTF-8 VL payloads.
///
/// Objects are packed [`MAX_HEAP_OBJECTS`] to a collection, so object `n` lives
/// in collection `n / MAX_HEAP_OBJECTS` at 1-based index
/// `n % MAX_HEAP_OBJECTS + 1`. [`patch_vl_refs`] and [`patch_vl_refs_masked`]
/// resolve references with that same rule, and [`stage_vl_elements`] and
/// [`build_attr_message`] write the matching indices.
pub(crate) fn build_global_heap_collections_from_bytes(objects: &[&[u8]]) -> Vec<Vec<u8>> {
    objects
        .chunks(MAX_HEAP_OBJECTS)
        .map(build_global_heap_collection_bytes)
        .collect()
}

/// Build one global heap collection holding `objects` (at most
/// [`MAX_HEAP_OBJECTS`] of them), assigning 1-based object indices in order.
/// Returns the serialized collection bytes.
fn build_global_heap_collection_bytes(objects: &[&[u8]]) -> Vec<u8> {
    debug_assert!(
        objects.len() <= MAX_HEAP_OBJECTS,
        "a collection's 2-byte object index cannot address more than {MAX_HEAP_OBJECTS} objects"
    );
    let length_size = 8usize;
    let header_size = 8 + length_size; // sig(4) + ver(1) + reserved(3) + collection_size

    // Calculate total size
    let mut obj_size_total = 0usize;
    for obj in objects {
        let obj_header = 8 + length_size; // index(2) + refcount(2) + reserved(4) + size
        let padded_data_len = (obj.len() + 7) & !7; // pad to 8 bytes
        obj_size_total += obj_header + padded_data_len;
    }
    obj_size_total += 8 + length_size; // free space marker (full object header size)
    let collection_size = header_size + obj_size_total;
    // The C HDF5 library enforces a minimum collection size of 4096 bytes.
    let min_collection_size = 4096;
    let padded_collection = ((collection_size.max(min_collection_size)) + 7) & !7;

    let mut buf = Vec::with_capacity(padded_collection);
    // Header
    buf.extend_from_slice(b"GCOL");
    buf.push(1); // version
    buf.extend_from_slice(&[0u8; 3]); // reserved
    buf.extend_from_slice(&(padded_collection as u64).to_le_bytes());

    // Objects (1-based indices)
    for (i, obj) in objects.iter().enumerate() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "1-based heap object index is written into the 2-byte heap-object index field"
        )]
        let index = (i + 1) as u16;
        buf.extend_from_slice(&index.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // ref_count
        buf.extend_from_slice(&[0u8; 4]); // reserved
        buf.extend_from_slice(&(obj.len() as u64).to_le_bytes());
        buf.extend_from_slice(obj);
        // Pad to 8-byte boundary
        let padded = (obj.len() + 7) & !7;
        for _ in obj.len()..padded {
            buf.push(0);
        }
    }

    // Free space marker (index 0): the C library uses this size as the total
    // skip distance from the start of the object (including its header), so
    // it must equal the remaining bytes in the collection from this point.
    let free_total_size = padded_collection - buf.len();
    buf.extend_from_slice(&0u16.to_le_bytes()); // index 0
    buf.extend_from_slice(&0u16.to_le_bytes()); // ref_count
    buf.extend_from_slice(&[0u8; 4]); // reserved
    buf.extend_from_slice(&(free_total_size as u64).to_le_bytes()); // size

    // Pad collection to full size
    buf.resize(padded_collection, 0);

    buf
}

/// Patch VL attribute references with the actual global heap collection
/// addresses. The raw_data contains VL references with placeholder addresses
/// (0), one per attribute element, each holding an object of the collection its
/// position selects (see [`build_global_heap_collections_from_bytes`]);
/// `collection_addresses` gives those collections' placed addresses in order.
pub(crate) fn patch_vl_refs(raw_data: &mut [u8], collection_addresses: &[u64]) {
    // Only a value that writes whole variable-length references reaches a heap,
    // and [`AttrValue::var_len_strings`] is what decides which values those are.
    // This is a screen on that decision, not a proof of it: it catches a value
    // whose element bytes are not a whole number of references — the division
    // below would then patch its *data*, or patch nothing and leave the
    // collection it caused to be written unreferenced — but a fixed-width value
    // whose bytes happen to be a multiple of 16, two `u64`s say, passes it.
    debug_assert_eq!(
        raw_data.len() % VL_REF_SIZE,
        0,
        "a value routed through the global heap must hold whole variable-length references"
    );
    let count = raw_data.len() / VL_REF_SIZE;
    for i in 0..count {
        let address = collection_addresses[i / MAX_HEAP_OBJECTS];
        let addr_offset = i * VL_REF_SIZE + 4; // skip sequence_length
        raw_data[addr_offset..addr_offset + 8].copy_from_slice(&address.to_le_bytes());
    }
}

/// A single element of a VL-string dataset being written: either a null
/// reference (no heap object) or a heap object carrying these exact bytes.
///
/// The two are distinct in the HDF5 model: a null reference reads back as a
/// null/empty element with no heap object, whereas a zero-length heap object
/// reads back as an empty string `""`. Carrying both lets a faithful rewrite
/// reproduce the source byte-for-byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VlStringElement {
    /// A null reference: length 0, undefined heap address, no heap object.
    Null,
    /// A heap object holding these exact bytes (possibly empty).
    Bytes(Vec<u8>),
}

/// 16-byte size of a single VL global-heap reference (offset_size = 8):
/// length(4) + collection address(8) + object index(4).
pub(crate) const VL_REF_SIZE: usize = 16;

/// Staged variable-length dataset/attribute element data: the element bytes
/// (carrying VL references with placeholder heap addresses) plus the global heap
/// collections holding the non-null elements' bytes.
///
/// The non-null elements' bytes become heap objects in order, packed
/// [`MAX_HEAP_OBJECTS`] to a collection, and each reference carries its
/// object's 1-based index within its own collection — matching
/// [`build_global_heap_collections_from_bytes`]. Null elements carry a zero
/// address and object index 0, and are never patched so they read back as null.
/// The element bytes this accompanies are returned beside it rather than held
/// here: they are the dataset's data, and the builder's `data` field is where
/// they live. Carrying them in both places cost a copy of every reference in the
/// dataset for a field nothing read afterwards (issue #228).
///
/// `Clone` is for the overwrite path, whose plan is built from a *borrowed*
/// staged edit: the staged set has to survive a refused commit intact (issue
/// #316), so the plan cannot move the staging out of it. The copy is of the
/// heap collections — the string bytes themselves, not the fixed-width element
/// references beside them — and is made once per staged variable-length
/// overwrite.
#[derive(Clone)]
pub(crate) struct VlStringStaging {
    /// The serialized global heap collections holding the non-null objects, in
    /// the order their objects appear. Empty when there are no such objects.
    pub collections: Vec<Vec<u8>>,
    /// Byte offset within the element bytes of each reference that names a heap
    /// object and so needs its address patched once the collections are placed,
    /// in the same order as those objects. Null references are absent: their
    /// address must stay zero so they read back as null.
    pub patch_offsets: Vec<usize>,
}

/// Stage the references and global-heap collections for a variable-length
/// dataset/attribute from its per-element byte payloads.
///
/// `element_size` is the byte width of the VL base type. A VL string's base type
/// is a single byte (`element_size == 1`), so the reference's stored `length`
/// equals the payload's byte count. A non-string VL sequence stores an element
/// *count* in that field, so the length written is `bytes.len() / element_size`,
/// while the heap object still holds the exact bytes.
pub(crate) fn stage_vl_elements(
    elements: &[VlStringElement],
    element_size: NonZeroUsize,
) -> (Vec<u8>, VlStringStaging) {
    stage_vl_payloads(
        elements.iter().map(|e| match e {
            VlStringElement::Null => None,
            VlStringElement::Bytes(bytes) => Some(bytes.as_slice()),
        }),
        element_size,
    )
}

/// [`stage_vl_elements`] over payloads the caller has not had to own.
///
/// A writer that already holds its strings — `with_vlen_strings` is handed
/// `&[&str]` — would otherwise copy every one of them into a
/// [`VlStringElement::Bytes`] first, which is the whole payload again in one
/// allocation per element: 4 MiB in 32,768 blocks to write 4 MiB of text
/// (issue #228). The bytes are copied once, into the heap collections, which is
/// where they have to end up.
pub(crate) fn stage_vl_payloads<'a>(
    payloads: impl ExactSizeIterator<Item = Option<&'a [u8]>>,
    element_size: NonZeroUsize,
) -> (Vec<u8>, VlStringStaging) {
    // Collect the non-null payloads in order; their positions become the heap
    // object indices, 1-based within each collection.
    let count = payloads.len();
    let mut objects: Vec<&[u8]> = Vec::new();
    let mut refs = Vec::with_capacity(count * VL_REF_SIZE);
    let mut patch_offsets = Vec::with_capacity(count);
    for element in payloads {
        match element {
            None => {
                // A null VL reference: HDF5 marks "no heap object" with a zero
                // heap address (`H5T__vlen_disk_isnull` tests addr == 0), not the
                // all-ones "undefined address" sentinel — the reference C library
                // rejects the latter as a bad heap index when reading.
                refs.extend_from_slice(&0u32.to_le_bytes()); // length 0
                refs.extend_from_slice(&0u64.to_le_bytes()); // null heap address
                refs.extend_from_slice(&0u32.to_le_bytes()); // object index 0
            }
            Some(bytes) => {
                patch_offsets.push(refs.len());
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "VL element length (element count) is written into the 4-byte \
                              length prefix of the variable-length reference"
                )]
                refs.extend_from_slice(&((bytes.len() / element_size) as u32).to_le_bytes());
                refs.extend_from_slice(&0u64.to_le_bytes()); // patched later
                // 1-based index within this object's own collection.
                let index = objects.len() % MAX_HEAP_OBJECTS + 1;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "1-based heap object index is written into the 4-byte object-index \
                              field of the variable-length reference"
                )]
                refs.extend_from_slice(&(index as u32).to_le_bytes());
                objects.push(bytes);
            }
        }
    }
    // A dataset of only null elements (or an empty dataset) references no heap
    // object, so it gets no collection at all — there is nothing to patch and a
    // 4096-byte empty GCOL would be dead weight.
    (
        refs,
        VlStringStaging {
            collections: build_global_heap_collections_from_bytes(&objects),
            patch_offsets,
        },
    )
}

/// Stage the global-heap collections for a dataset whose datatype merely
/// *contains* variable-length members — a compound with a VL member, or an array
/// of such compounds — rather than being variable-length itself.
///
/// `raw` is the dataset's element bytes as read from the source, and `offsets`
/// gives the byte offset of every embedded VL reference within `raw`, in the same
/// order as `elements`. Each reference is rewritten in place: a null element is
/// zeroed outright, and a heap-backed one keeps the source's `length` field
/// (which counts base-type elements, whose width varies per member) while
/// gaining the destination's object index. The addresses stay at zero until
/// [`patch_vl_refs_masked`] fills them in, exactly as for a top-level VL dataset.
pub(crate) fn stage_embedded_vl_elements(
    mut raw: Vec<u8>,
    offsets: &[usize],
    elements: &[VlStringElement],
) -> (Vec<u8>, VlStringStaging) {
    debug_assert_eq!(
        offsets.len(),
        elements.len(),
        "one staged payload per embedded variable-length reference"
    );
    let mut objects: Vec<&[u8]> = Vec::new();
    let mut patch_offsets = Vec::with_capacity(elements.len());
    for (&offset, element) in offsets.iter().zip(elements) {
        let slot = &mut raw[offset..offset + VL_REF_SIZE];
        match element {
            VlStringElement::Null => slot.fill(0),
            VlStringElement::Bytes(bytes) => {
                patch_offsets.push(offset);
                // Leave the source's element-count `length` in bytes 0..4 alone,
                // and blank the address so an unpatched reference is visibly null
                // rather than a stale source address.
                slot[4..12].fill(0);
                let index = objects.len() % MAX_HEAP_OBJECTS + 1;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "1-based heap object index is written into the 4-byte object-index \
                              field of the variable-length reference"
                )]
                slot[12..16].copy_from_slice(&(index as u32).to_le_bytes());
                objects.push(bytes);
            }
        }
    }
    (
        raw,
        VlStringStaging {
            collections: build_global_heap_collections_from_bytes(&objects),
            patch_offsets,
        },
    )
}

/// Patch the heap address of each VL reference named by `patch_offsets`, leaving
/// null references (which are absent from that list) with their zero address so
/// they read back as null. Mirrors [`patch_vl_refs`], but because only
/// heap-backed references are listed, the collection a reference resolves to is
/// selected by its *object* position rather than its element position — and the
/// references need not sit at a fixed stride, which is what lets a compound with
/// variable-length members share this path. `collection_addresses` holds the
/// placed addresses of [`VlStringStaging::collections`], in order.
pub(crate) fn patch_vl_refs_masked(
    raw_data: &mut [u8],
    patch_offsets: &[usize],
    collection_addresses: &[u64],
) {
    for (object_ordinal, &offset) in patch_offsets.iter().enumerate() {
        let address = collection_addresses[object_ordinal / MAX_HEAP_OBJECTS];
        let addr_offset = offset + 4; // skip sequence_length
        raw_data[addr_offset..addr_offset + 8].copy_from_slice(&address.to_le_bytes());
    }
}

/// `STRSIZE` for a fixed-width string datatype whose longest value is `len`
/// bytes, which is never zero.
///
/// HDF5 requires a string datatype of at least one byte. libhdf5 rejects a
/// zero-size one with "invalid datatype size" — and it fails while *iterating*
/// the object's attributes, so a single empty-string attribute makes every
/// attribute on that object unreadable to the C library, not just that one.
///
/// An empty string is therefore stored as one padding byte, which reads back as
/// the empty string under any of the three padding rules. Storing it needs no
/// refusal: the value is representable, it was only the datatype that was not.
///
/// The upper end is the datatype message's own 4-byte size field: no fixed-width
/// string element can be wider than [`u32::MAX`], so a longer `len` saturates
/// there rather than wrapping to a small width whose elements would each store a
/// truncated prefix. Saturating is not itself an answer for such a value, only a
/// refusal to corrupt one; what refuses it differs by caller. A dataset's values
/// then meet [`encode_fixed_strings`], which finds each one longer than the width
/// and reports it. An attribute has no such channel — every `set_attr` is
/// infallible — so it stores the first `u32::MAX` bytes and declares that width,
/// which [`pad_fixed_strings`] is where it happens. The compact-attribute cap of
/// [`OBJECT_HEADER_MESSAGE_MAX`](crate::OBJECT_HEADER_MESSAGE_MAX) does not
/// refuse the value first: an attribute past it moves to dense storage, and a
/// 70,000-byte fixed string writes and reads back at its declared width.
fn fixed_string_size(len: usize) -> NonZeroU32 {
    // The `unwrap_or` is the empty-string rule above: zero bytes still take one.
    NonZeroU32::new(u32::try_from(len).unwrap_or(u32::MAX)).unwrap_or(NonZeroU32::MIN)
}

/// `STRSIZE` for a fixed-width string built from `values`: wide enough for the
/// longest of them, and never zero.
///
/// Shared with the attribute side so the same values written as an
/// [`AttrValue::AsciiStringArray`] and as a dataset declare the same width, and
/// generic over the string type because the attribute variants own theirs while
/// the dataset entry points borrow the caller's.
fn derived_string_width<S: AsRef<str>>(values: &[S]) -> NonZeroU32 {
    fixed_string_size(values.iter().map(|s| s.as_ref().len()).max().unwrap_or(0))
}

/// Whether a declared width of `width` holds every one of `values`.
///
/// A value longer than `width` is refused rather than truncated. A stored prefix
/// reads back as a value the caller never wrote, and reads back without error,
/// so nothing downstream could tell it apart from real data.
///
/// Reached through [`checked_width`], which is the whole rule — this plus the
/// zero-width refusal — and is what both places a width can be declared go
/// through, so a dataset and an attribute cannot disagree about which values are
/// storable: a dataset checks as its values are staged
/// ([`DatasetBuilder::with_ascii_strings_sized`](DatasetBuilder::with_ascii_strings_sized)
/// and its siblings), an attribute when the value is constructed
/// ([`AttrValue::ascii_string_sized`] and its siblings).
fn check_fixed_width<S: AsRef<str>>(values: &[S], width: NonZeroU32) -> Result<(), FormatError> {
    for (index, value) in values.iter().enumerate() {
        let len = value.as_ref().len();
        if len > width.get() as usize {
            return Err(FormatError::FixedStringTooLong {
                index,
                len,
                width: width.get(),
            });
        }
    }
    Ok(())
}

/// The element bytes of a fixed-width string value, and the `STRSIZE` they were
/// laid out to: each value's bytes zero-padded on the right, which is what
/// [`StringPadding::NullPad`] declares on the wire.
///
/// `min_width` is a floor, not the answer: the width used is that or the longest
/// value's own length, whichever is larger. Passing [`NonZeroU32::MIN`] is
/// therefore how a caller asks for the *derived* width, and passing a declared
/// one asks for that width unless a value would not fit it — which no caller
/// does, since a declared width has been through [`checked_width`] first, and
/// that is what the sized [`AttrValue`] variants are sealed for.
///
/// Returning the width beside the bytes is what stops the datatype message and
/// the elements it describes from disagreeing about it.
fn pad_fixed_strings<S: AsRef<str>>(values: &[S], min_width: NonZeroU32) -> (Vec<u8>, NonZeroU32) {
    let width = min_width.max(derived_string_width(values));
    let width_bytes = width.get() as usize;
    let mut raw = Vec::with_capacity(values.len().saturating_mul(width_bytes));
    for value in values {
        // The one value `width` does not hold is one this format cannot describe
        // at all: [`derived_string_width`] saturates at `u32::MAX`, the widest a
        // datatype message's size field can state, so anything past that is
        // stored as its first `u32::MAX` bytes. That truncation is what the
        // saturation has always meant here, and the alternative is arithmetic
        // that underflows on caller data no `set_attr` signature can refuse.
        let bytes = value.as_ref().as_bytes();
        let bytes = &bytes[..bytes.len().min(width_bytes)];
        raw.extend_from_slice(bytes);
        raw.resize(raw.len() + (width_bytes - bytes.len()), 0);
    }
    (raw, width)
}

/// The raw element bytes of a fixed-width string dataset of width `width`,
/// refusing a value that width cannot hold.
///
/// This is where the *derived* dataset entry points get their only check: their
/// width saturates at `u32::MAX` for a value no datatype message can describe,
/// and a dataset, unlike an attribute, has a `Result` to report that on. The
/// declared entry points have been through [`checked_width`] already, so for
/// them this check is a second look that cannot disagree with the first — the
/// same function, over the same values.
fn encode_fixed_strings<S: AsRef<str>>(
    values: &[S],
    width: NonZeroU32,
) -> Result<Vec<u8>, FormatError> {
    check_fixed_width(values, width)?;
    Ok(pad_fixed_strings(values, width).0)
}

pub(crate) fn scalar_ds() -> Dataspace {
    Dataspace {
        space_type: DataspaceType::Scalar,
        rank: 0,
        dimensions: vec![],
        max_dimensions: None,
    }
}

pub(crate) fn simple_1d(n: u64) -> Dataspace {
    Dataspace {
        space_type: DataspaceType::Simple,
        rank: 1,
        dimensions: vec![n],
        max_dimensions: None,
    }
}

// ---- Attribute values ----

/// Convenient attribute values for the write API.
///
/// Each numeric variant names the width it is stored at, and a value written
/// from one reads back as the same variant: an `I16` attribute is two bytes on
/// disk and arrives as `I16`, not widened to `I64` (issue #350), and an `F32`
/// is four and arrives as `F32` (issue #354). The accessors below read any
/// integer as `i64`/`u64` and any float as `f64`, so code that only wants the
/// number need not enumerate the widths.
///
/// A fixed-width string keeps its width the same way. The plain string variants
/// size the datatype to the content, which is what a caller writing a value
/// wants; the `*Sized` ones declare a width instead, for a slot that has to
/// stay the same size across rewrites (issue #359). The accessors span both, so
/// code that only wants the text need not know which it was handed.
///
/// A variable-length string keeps its datatype the same way. The four
/// [`VarLenString`](AttrValue::VarLenString) and
/// [`VarLenAsciiString`](AttrValue::VarLenAsciiString) variants write the
/// standard `H5T_STRING` with `STRSIZE = H5T_VARIABLE`, which is what h5py and
/// the reference C library write and read;
/// [`VarLenAsciiCharArray`](AttrValue::VarLenAsciiCharArray) writes MATLAB's sequence
/// of one-byte strings over identical element bytes (issue #383). Both keep
/// their strings in a global heap collection rather than in the attribute
/// message.
///
/// Non-exhaustive: variants are added as this crate supports more attribute
/// datatypes, so match a read-back value with a `_` arm. Constructing the
/// variants below is unaffected.
///
/// # HDF5 encodings
///
/// | Variant | HDF5 encoding |
/// |---|---|
/// | [`F32`](Self::F32) / [`F32Array`](Self::F32Array) | 32-bit float scalar / array |
/// | [`F64`](Self::F64) / [`F64Array`](Self::F64Array) | 64-bit float scalar / array |
/// | [`I8`](Self::I8) / [`I16`](Self::I16) / [`I32`](Self::I32) / [`I64`](Self::I64) | Signed integer scalar, at that width |
/// | [`I8Array`](Self::I8Array) / [`I16Array`](Self::I16Array) / [`I32Array`](Self::I32Array) / [`I64Array`](Self::I64Array) | Signed integer array, at that width |
/// | [`U8`](Self::U8) / [`U16`](Self::U16) / [`U32`](Self::U32) / [`U64`](Self::U64) | Unsigned integer scalar, at that width |
/// | [`U8Array`](Self::U8Array) / [`U16Array`](Self::U16Array) / [`U32Array`](Self::U32Array) / [`U64Array`](Self::U64Array) | Unsigned integer array, at that width |
/// | [`String`](Self::String) | UTF-8 null-padded string |
/// | [`StringArray`](Self::StringArray) | UTF-8 null-padded string array |
/// | [`AsciiString`](Self::AsciiString) | Fixed-width ASCII string |
/// | [`AsciiStringArray`](Self::AsciiStringArray) | Array of fixed-width ASCII strings (null-padded to the longest element) |
/// | [`VarLenAsciiCharArray`](Self::VarLenAsciiCharArray) | MATLAB's variable-length ASCII string array, a VLEN sequence of one-byte strings (stored in a global heap collection) |
/// | [`VarLenString`](Self::VarLenString) / [`VarLenStringArray`](Self::VarLenStringArray) | Variable-length UTF-8 string, scalar or array: `H5T_STRING` with `STRSIZE = H5T_VARIABLE` (stored in a global heap collection) |
/// | [`VarLenAsciiString`](Self::VarLenAsciiString) / [`VarLenAsciiStringArray`](Self::VarLenAsciiStringArray) | The same, with `CSET = ASCII` |
/// | [`StringSized`](Self::StringSized) / [`AsciiStringSized`](Self::AsciiStringSized) (and their `*ArraySized` forms) | Fixed-width string of a declared width, built by [`string_sized`](Self::string_sized) / [`ascii_string_sized`](Self::ascii_string_sized) and their array siblings |
///
/// [`AsciiStringArray`](Self::AsciiStringArray) and
/// [`VarLenAsciiCharArray`](Self::VarLenAsciiCharArray) exist for MATLAB interoperability (the
/// `MATLAB_fields` pattern). The [`VarLenString`](Self::VarLenString) family writes the standard
/// variable-length string datatype, which is the one h5py and the C library write and read.
/// See the [groups and attributes guide](crate::_guide::groups_attributes).
///
/// # Reading attributes back
///
/// Reading attributes yields the same enum, a `HashMap<String, AttrValue>` from
/// [`attrs`](crate::Dataset::attrs). What the reader normalizes is the layout inside a value's
/// width: every numeric variant is little-endian at its full precision and, for a float, in IEEE
/// 754 layout, so a big-endian attribute reads correctly and would be rewritten in this crate's own
/// layout. An integer width with no Rust integer of its own, 3 bytes say, comes back as
/// [`I64`](Self::I64) / [`U64`](Self::U64). An enumeration attribute decodes through its integer
/// base type, the same view the numeric readers take of an enum dataset, so its codes arrive and
/// its member names do not, which is how an h5py `np.bool_` attribute, stored as `enum[FALSE,
/// TRUE]`, reads back as `0` or `1`. [`repack`](crate::repack()) does not go through `AttrValue`,
/// so an attribute keeps whatever encoding it had when copied.
///
/// [`attr_datatypes`](crate::Dataset::attr_datatypes) reports what that normalization drops: the
/// [`Datatype`] of every attribute, including the ones [`attrs`](crate::Dataset::attrs) omits for
/// having no `AttrValue` at all. It reports for an attribute what
/// [`Dataset::datatype`](crate::Dataset::datatype) reports for a dataset, and it is what identifies
/// a `np.bool_` attribute as boolean, where `AttrValue` gives an ordinary one-byte integer. As
/// [`Dataset::datatype`](crate::Dataset::datatype) does, it resolves a committed (shared) datatype
/// to the type it refers to, though neither reports that type's name, and neither channel exposes
/// an attribute's rank. See the
/// [groups and attributes guide](crate::_guide::groups_attributes#reading-an-attributes-datatype).
///
/// # Examples
///
/// ```rust
/// use hdf5_pure::{FileBuilder, AttrValue};
///
/// let mut builder = FileBuilder::new();
/// builder.set_attr("version", AttrValue::I64(2));
/// builder.set_attr("unit", AttrValue::AsciiString("m/s".into()));
///
/// // A slot of a chosen width, holding a shorter value: `H5T_C_S1` plus
/// // `H5Tset_size(64)`. Writing it again at the same width leaves the slot the
/// // size it was, where `AsciiString` would resize it to the new content.
/// builder.set_attr("label", AttrValue::ascii_string_sized("ok", 64)?);
/// # let file = hdf5_pure::File::from_bytes(builder.finish()?)?;
/// # let attrs = file.root().attrs()?;
/// # assert_eq!(attrs["version"], AttrValue::I64(2));
/// # assert_eq!(attrs["label"].as_str(), Some("ok"));
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AttrValue {
    F32(f32),
    F32Array(Vec<f32>),
    F64(f64),
    F64Array(Vec<f64>),
    I8(i8),
    I8Array(Vec<i8>),
    I16(i16),
    I16Array(Vec<i16>),
    I32(i32),
    I32Array(Vec<i32>),
    I64(i64),
    I64Array(Vec<i64>),
    U8(u8),
    U8Array(Vec<u8>),
    U16(u16),
    U16Array(Vec<u16>),
    U32(u32),
    U32Array(Vec<u32>),
    U64(u64),
    /// Unsigned 64-bit integer array. Distinct from
    /// [`I64Array`](AttrValue::I64Array) because a value above [`i64::MAX`] has
    /// no `i64` to be stored as.
    U64Array(Vec<u64>),
    /// UTF-8 string attribute, sized to its own content (an empty one still
    /// takes one byte, the narrowest datatype HDF5 allows).
    String(String),
    /// UTF-8 string attribute of a width declared rather than derived, built by
    /// [`string_sized`](AttrValue::string_sized). See
    /// [`AsciiStringSized`](AttrValue::AsciiStringSized), whose only difference
    /// is the charset.
    #[non_exhaustive]
    StringSized {
        /// The string, without the padding that brings it up to `width`.
        value: String,
        /// The declared `STRSIZE`, in bytes.
        width: NonZeroU32,
    },
    /// Array of fixed-width UTF-8 strings, null-padded to the longest element.
    StringArray(Vec<String>),
    /// Array of fixed-width UTF-8 strings of a width declared rather than
    /// derived, built by
    /// [`string_array_sized`](AttrValue::string_array_sized). The array form of
    /// [`StringSized`](AttrValue::StringSized), sealed for the same reason.
    #[non_exhaustive]
    StringArraySized {
        /// The strings, each without the padding that brings it up to `width`.
        values: Vec<String>,
        /// The declared `STRSIZE`, in bytes, shared by every element.
        width: NonZeroU32,
    },
    /// Fixed-width ASCII string attribute (charset = ASCII), sized to its own
    /// content (an empty one still takes one byte).
    AsciiString(String),
    /// Fixed-width ASCII string attribute of a width declared rather than
    /// derived, built by
    /// [`ascii_string_sized`](AttrValue::ascii_string_sized).
    ///
    /// This is the `H5T_C_S1` + `H5Tset_size(N)` idiom: a slot of a chosen
    /// width, holding a shorter string with the rest null-padded, which keeps
    /// that width when the attribute is written again (issue #359). The plain
    /// [`AsciiString`](AttrValue::AsciiString) takes the content's own length
    /// instead, so rewriting a 64-byte slot with a 2-byte string shrinks it.
    ///
    /// Read back as this variant only when the stored width is *wider* than the
    /// content implies; a slot sized to its content reads back as
    /// [`AsciiString`](AttrValue::AsciiString), which writes the same bytes.
    ///
    /// Sealed: build it with
    /// [`ascii_string_sized`](AttrValue::ascii_string_sized), which is where a
    /// value the width cannot hold is refused. Reading the fields, and matching
    /// with a `..`, are unaffected.
    #[non_exhaustive]
    AsciiStringSized {
        /// The string, without the padding that brings it up to `width`.
        value: String,
        /// The declared `STRSIZE`, in bytes.
        width: NonZeroU32,
    },
    /// Array of fixed-width ASCII strings (null-padded to the longest element).
    /// Compatible with MATLAB `MATLAB_fields` and matio.
    AsciiStringArray(Vec<String>),
    /// Array of fixed-width ASCII strings of a width declared rather than
    /// derived, built by
    /// [`ascii_string_array_sized`](AttrValue::ascii_string_array_sized). The
    /// array form of [`AsciiStringSized`](AttrValue::AsciiStringSized), sealed
    /// for the same reason.
    #[non_exhaustive]
    AsciiStringArraySized {
        /// The strings, each without the padding that brings it up to `width`.
        values: Vec<String>,
        /// The declared `STRSIZE`, in bytes, shared by every element.
        width: NonZeroU32,
    },
    /// Array of variable-length ASCII **char sequences** in MATLAB's shape:
    /// `H5T_VLEN { H5T_STRING { STRSIZE = 1, NULLTERM, ASCII } }`. Each element
    /// is a sequence of one-byte strings rather than one variable-length string,
    /// and MATLAB v7.3 and matio expect `MATLAB_fields` and its neighbours in
    /// exactly this datatype.
    ///
    /// For the standard variable-length string — a `H5T_STRING` of variable size,
    /// which is what h5py and the reference C library write — reach for
    /// [`VarLenAsciiStringArray`](AttrValue::VarLenAsciiStringArray) instead.
    /// Its element bytes are identical to these; only the datatype differs. Both
    /// need a global heap collection in the file.
    VarLenAsciiCharArray(Vec<String>),
    /// A variable-length UTF-8 string: `H5T_STRING` with
    /// `STRSIZE = H5T_VARIABLE`, `NULLTERM`, `CSET = UTF-8`.
    ///
    /// The standard encoding — what h5py, the reference C library and this
    /// crate's own `DatasetBuilder::with_vlen_strings` write, and what h5py
    /// hands back as a `str`. The value lives in a global heap collection, and
    /// the attribute holds one 16-byte reference to it.
    ///
    /// [`String`](AttrValue::String) writes a *fixed*-width slot sized to the
    /// value instead, which readers generally accept too; reach for this one
    /// when the file's consumer expects a variable-length string specifically.
    VarLenString(String),
    /// Array of variable-length UTF-8 strings. The one-dimensional form of
    /// [`VarLenString`](AttrValue::VarLenString), with one heap reference per
    /// element.
    VarLenStringArray(Vec<String>),
    /// A variable-length ASCII string: `H5T_STRING` with
    /// `STRSIZE = H5T_VARIABLE`, `NULLTERM`, `CSET = ASCII` — the
    /// `H5Tcopy(H5T_C_S1)` plus `H5Tset_size(H5T_VARIABLE)` idiom, which h5py
    /// hands back as `bytes`.
    ///
    /// The charset is the only difference from
    /// [`VarLenString`](AttrValue::VarLenString): the bytes written are the
    /// same, and this crate does not check that they are ASCII, exactly as
    /// [`AsciiString`](AttrValue::AsciiString) does not.
    VarLenAsciiString(String),
    /// Array of variable-length ASCII strings, in the standard
    /// `H5T_STRING`/`STRSIZE = H5T_VARIABLE` datatype. The one-dimensional form
    /// of [`VarLenAsciiString`](AttrValue::VarLenAsciiString).
    ///
    /// [`VarLenAsciiCharArray`](AttrValue::VarLenAsciiCharArray) writes MATLAB's
    /// sequence-of-one-byte-strings shape over the same element bytes.
    VarLenAsciiStringArray(Vec<String>),
}

/// Constructors for the variants that declare a string width rather than
/// deriving one from the value.
///
/// Every other variant is written as a literal. These four are functions
/// because a declared width is the one thing an attribute value can be asked
/// for and be unable to hold, and refusing here rather than at the write is
/// what lets every `set_attr` entry point stay infallible. That is also why the
/// variants they build are sealed: a checked constructor is no invariant at all
/// if the same value can be written out by hand beside it.
impl AttrValue {
    /// A fixed-width ASCII string attribute `width` bytes wide.
    ///
    /// The datatype is `H5T_STRING { STRSIZE = width, NULLPAD, ASCII }` — the
    /// `H5T_C_S1` plus `H5Tset_size(width)` slot — and the value is null-padded
    /// out to it. Writing the attribute again from another value of the same
    /// width leaves the slot the size it was, where
    /// [`AsciiString`](AttrValue::AsciiString) resizes it to whatever the new
    /// content needs.
    ///
    /// A value longer than `width` is refused with
    /// [`FormatError::FixedStringTooLong`] rather than stored as a prefix, and a
    /// `width` of zero with [`FormatError::ZeroFixedStringWidth`], since no HDF5
    /// string datatype may be zero bytes wide. `width` counts bytes.
    ///
    /// ```
    /// use hdf5_pure::{AttrValue, File, FileBuilder, FormatError};
    ///
    /// let mut fb = FileBuilder::new();
    /// fb.create_dataset("reading")
    ///     .with_i32_data(&[1])
    ///     .set_attr("units", AttrValue::ascii_string_sized("ok", 64).unwrap());
    /// let file = File::from_bytes(fb.finish().unwrap()).unwrap();
    ///
    /// let attrs = file.dataset("reading").unwrap().attrs().unwrap();
    /// assert_eq!(attrs["units"].as_str(), Some("ok"));
    /// assert!(matches!(
    ///     attrs["units"],
    ///     AttrValue::AsciiStringSized { width, .. } if width.get() == 64
    /// ));
    ///
    /// // A value the declared width cannot hold is refused, not truncated.
    /// assert!(matches!(
    ///     AttrValue::ascii_string_sized("far too long", 2),
    ///     Err(FormatError::FixedStringTooLong { index: 0, len: 12, width: 2 })
    /// ));
    /// ```
    pub fn ascii_string_sized(value: impl Into<String>, width: u32) -> Result<Self, FormatError> {
        let value = value.into();
        let width = checked_width(core::slice::from_ref(&value), width)?;
        Ok(Self::AsciiStringSized { value, width })
    }

    /// An array of fixed-width ASCII strings, every element `width` bytes wide.
    ///
    /// The array form of
    /// [`ascii_string_sized`](AttrValue::ascii_string_sized), refusing the same
    /// two ways. [`FormatError::FixedStringTooLong`] names the position of the
    /// element that did not fit.
    pub fn ascii_string_array_sized(values: Vec<String>, width: u32) -> Result<Self, FormatError> {
        let width = checked_width(&values, width)?;
        Ok(Self::AsciiStringArraySized { values, width })
    }

    /// A fixed-width UTF-8 string attribute `width` bytes wide.
    ///
    /// The UTF-8 counterpart of
    /// [`ascii_string_sized`](AttrValue::ascii_string_sized): the same encoding
    /// under a different charset bit, refusing the same two ways. `width`
    /// counts *bytes*, so a value with multi-byte characters takes more of it
    /// than its character count suggests, and a value is measured whole — this
    /// never splits one across the width boundary, because it never truncates
    /// at all.
    pub fn string_sized(value: impl Into<String>, width: u32) -> Result<Self, FormatError> {
        let value = value.into();
        let width = checked_width(core::slice::from_ref(&value), width)?;
        Ok(Self::StringSized { value, width })
    }

    /// An array of fixed-width UTF-8 strings, every element `width` bytes wide.
    ///
    /// The array form of [`string_sized`](AttrValue::string_sized), refusing the
    /// same two ways.
    pub fn string_array_sized(values: Vec<String>, width: u32) -> Result<Self, FormatError> {
        let width = checked_width(&values, width)?;
        Ok(Self::StringArraySized { values, width })
    }
}

/// A width a caller declared, accepted only once it is a width at all and holds
/// every one of `values`.
fn checked_width<S: AsRef<str>>(values: &[S], width: u32) -> Result<NonZeroU32, FormatError> {
    let width = NonZeroU32::new(width).ok_or(FormatError::ZeroFixedStringWidth)?;
    check_fixed_width(values, width)?;
    Ok(width)
}

/// The width to *declare* for a fixed-width string read out of a file, or
/// `None` when the plain variant already reproduces it.
///
/// The plain variants write the width [`derived_string_width`] gives, so a
/// stored width equal to that one is exactly what they reproduce. Reporting the
/// sized variant only for a *wider* slot is what keeps a value written from
/// [`AttrValue::AsciiString`] reading back as `AsciiString`, while a padded slot
/// — the case that had no representation before issue #359 — arrives carrying
/// the width it would otherwise lose.
///
/// A slot *narrower* than its decoded text takes the plain variant too, and the
/// comparison is `>` rather than `!=` for exactly that case: the decoder is
/// lossy, and one invalid byte becomes a three-byte `U+FFFD`, so a two-byte
/// Latin-1 slot can decode to four bytes of text. Declaring the stored width
/// over that text would build a sized variant holding a value it cannot fit —
/// the pair [`checked_width`] refuses and the seal exists to prevent — and the
/// `width` field would be a claim about the bytes beside it that is simply
/// false.
fn declared_width<S: AsRef<str>>(values: &[S], stored: u32) -> Option<NonZeroU32> {
    let stored = NonZeroU32::new(stored)?;
    (stored > derived_string_width(values)).then_some(stored)
}

/// The [`AttrValue`] for a scalar fixed-width string attribute read out of a
/// file, in the variant that preserves the width it was stored at.
pub(crate) fn decoded_fixed_string(value: String, width: u32, charset: &CharacterSet) -> AttrValue {
    let ascii = *charset == CharacterSet::Ascii;
    match (declared_width(core::slice::from_ref(&value), width), ascii) {
        (Some(width), true) => AttrValue::AsciiStringSized { value, width },
        (Some(width), false) => AttrValue::StringSized { value, width },
        (None, true) => AttrValue::AsciiString(value),
        (None, false) => AttrValue::String(value),
    }
}

/// The [`AttrValue`] for an array of fixed-width strings read out of a file, in
/// the variant that preserves the width it was stored at.
pub(crate) fn decoded_fixed_string_array(
    values: Vec<String>,
    width: u32,
    charset: &CharacterSet,
) -> AttrValue {
    let ascii = *charset == CharacterSet::Ascii;
    match (declared_width(&values, width), ascii) {
        (Some(width), true) => AttrValue::AsciiStringArraySized { values, width },
        (Some(width), false) => AttrValue::StringArraySized { values, width },
        (None, true) => AttrValue::AsciiStringArray(values),
        (None, false) => AttrValue::StringArray(values),
    }
}

/// Accessors that read a value without matching on its variant.
///
/// One logical value has several representations here. A single string is a
/// [`String`](AttrValue::String) or an [`AsciiString`](AttrValue::AsciiString)
/// depending on charset, and a one-element array of either carries the same
/// thing; an integer spans four widths at two signednesses. Code that only wants
/// the value should not have to enumerate those, so each accessor spans every
/// variant that can carry the shape it names and returns `None` for the rest.
///
/// The prefix states the cost. `as_*` borrows or copies; `to_*` allocates,
/// which the numeric plurals must do because the narrower widths have no
/// `&[i64]` or `&[f64]` view to hand out.
///
/// ```
/// use hdf5_pure::AttrValue;
///
/// assert_eq!(AttrValue::AsciiString("double".into()).as_str(), Some("double"));
/// assert_eq!(AttrValue::StringArray(vec!["double".into()]).as_str(), Some("double"));
/// assert_eq!(AttrValue::F64(1.5).as_str(), None);
///
/// // A scalar reads as one element, without allocating.
/// let one = AttrValue::String("m/s".into());
/// assert_eq!(one.as_strings().unwrap(), ["m/s"]);
///
/// assert_eq!(AttrValue::I32(-7).to_i64s(), Some(vec![-7]));
/// ```
impl AttrValue {
    /// The value as one string, when it holds exactly one.
    ///
    /// Spans both charsets, fixed-width and variable-length, scalar or one
    /// element.
    /// Returns `None` for a non-string value, or for an array whose length is
    /// not 1.
    pub fn as_str(&self) -> Option<&str> {
        // Exactly [`as_strings`](AttrValue::as_strings) narrowed to one element,
        // rather than a second list of every string variant to keep in step with
        // it: a scalar is viewed there as a one-element slice, so both shapes
        // land in the same arm here.
        match self.as_strings()? {
            [one] => Some(one),
            _ => None,
        }
    }

    /// Every string the value holds, with a scalar reading as one element.
    ///
    /// Borrows: a scalar is viewed as a one-element slice rather than copied.
    /// Returns `None` for a non-string value. An empty array yields an empty
    /// slice, which is distinct from `None`.
    pub fn as_strings(&self) -> Option<&[String]> {
        match self {
            Self::String(s)
            | Self::AsciiString(s)
            | Self::StringSized { value: s, .. }
            | Self::AsciiStringSized { value: s, .. }
            | Self::VarLenString(s)
            | Self::VarLenAsciiString(s) => Some(core::slice::from_ref(s)),
            Self::StringArray(v)
            | Self::AsciiStringArray(v)
            | Self::VarLenAsciiCharArray(v)
            | Self::VarLenStringArray(v)
            | Self::VarLenAsciiStringArray(v)
            | Self::StringArraySized { values: v, .. }
            | Self::AsciiStringArraySized { values: v, .. } => Some(v),
            _ => None,
        }
    }

    /// The strings this value stores in a global heap collection rather than
    /// inline, or `None` for every value written into the attribute message
    /// itself.
    ///
    /// This is the one predicate that decides whether writing a value needs a
    /// heap collection built and its placeholder addresses patched afterwards.
    /// Every writer asks it rather than naming variants of its own, so a
    /// variable-length variant added later cannot reach a writer that would
    /// emit its placeholder addresses as though they were real ones.
    pub(crate) fn var_len_strings(&self) -> Option<&[String]> {
        match self {
            Self::VarLenAsciiCharArray(v)
            | Self::VarLenStringArray(v)
            | Self::VarLenAsciiStringArray(v) => Some(v),
            Self::VarLenString(s) | Self::VarLenAsciiString(s) => Some(core::slice::from_ref(s)),
            _ => None,
        }
    }

    /// The value as one `i64`, when it holds exactly one integer.
    ///
    /// Narrower signed and unsigned widths widen exactly. A value above
    /// [`i64::MAX`] does not fit and yields `None` rather than a wrapped
    /// negative; [`as_u64`](AttrValue::as_u64) reads it.
    pub fn as_i64(&self) -> Option<i64> {
        self.single_int()
    }

    /// The value as one `u64`, when it holds exactly one integer.
    ///
    /// Unsigned widths widen exactly. A negative value has no `u64` and yields
    /// `None`; [`as_i64`](AttrValue::as_i64) reads it.
    pub fn as_u64(&self) -> Option<u64> {
        self.single_int()
    }

    /// Every integer the value holds as `i64`, with a scalar reading as one
    /// element.
    ///
    /// Returns `None` for a non-integer value, and for an unsigned value with
    /// **any** element above [`i64::MAX`] — the range rule is per element, not
    /// per variant, so a wrapped negative is never handed back. Read those
    /// through [`to_u64s`](AttrValue::to_u64s).
    pub fn to_i64s(&self) -> Option<Vec<i64>> {
        self.int_elements()
    }

    /// Every integer the value holds as `u64`, with a scalar reading as one
    /// element.
    ///
    /// Returns `None` for a non-integer value, and for a signed value with any
    /// negative element.
    pub fn to_u64s(&self) -> Option<Vec<u64>> {
        self.int_elements()
    }

    /// The one integer this value holds, as `T`, or `None` unless it holds
    /// exactly one — a scalar, or an array of length 1.
    ///
    /// Every width goes through `i128`, which holds [`i64::MIN`] and
    /// [`u64::MAX`] alike, so the widening here is always exact and the only
    /// narrowing is the caller's own conversion to `T`. That is what lets one
    /// list of variants serve both [`as_i64`](AttrValue::as_i64) and
    /// [`as_u64`](AttrValue::as_u64): a width added to the enum reaches both or
    /// neither.
    fn single_int<T: TryFrom<i128>>(&self) -> Option<T> {
        let one: i128 = match self {
            Self::I8(v) => (*v).into(),
            Self::I16(v) => (*v).into(),
            Self::I32(v) => (*v).into(),
            Self::I64(v) => (*v).into(),
            Self::U8(v) => (*v).into(),
            Self::U16(v) => (*v).into(),
            Self::U32(v) => (*v).into(),
            Self::U64(v) => (*v).into(),
            Self::I8Array(v) if v.len() == 1 => v[0].into(),
            Self::I16Array(v) if v.len() == 1 => v[0].into(),
            Self::I32Array(v) if v.len() == 1 => v[0].into(),
            Self::I64Array(v) if v.len() == 1 => v[0].into(),
            Self::U8Array(v) if v.len() == 1 => v[0].into(),
            Self::U16Array(v) if v.len() == 1 => v[0].into(),
            Self::U32Array(v) if v.len() == 1 => v[0].into(),
            Self::U64Array(v) if v.len() == 1 => v[0].into(),
            _ => return None,
        };
        T::try_from(one).ok()
    }

    /// Every integer this value holds, as `T`, with a scalar reading as one
    /// element. `None` for a non-integer value, or for any element `T` cannot
    /// hold — the range rule is per element, so a whole array is refused rather
    /// than one of its values silently wrapping.
    ///
    /// Widths go through `i128` for the reason [`single_int`](AttrValue::single_int)
    /// gives, and each element is converted once, into the vector handed back.
    fn int_elements<T: TryFrom<i128>>(&self) -> Option<Vec<T>> {
        match self {
            Self::I8Array(v) => many_ints(v),
            Self::I16Array(v) => many_ints(v),
            Self::I32Array(v) => many_ints(v),
            Self::I64Array(v) => many_ints(v),
            Self::U8Array(v) => many_ints(v),
            Self::U16Array(v) => many_ints(v),
            Self::U32Array(v) => many_ints(v),
            Self::U64Array(v) => many_ints(v),
            // Everything else holds at most one integer, and `single_int` is
            // already that list; a non-integer value refuses there.
            _ => Some(vec![self.single_int()?]),
        }
    }

    /// The value as one `f64`, when it holds exactly one float.
    ///
    /// Integer variants are not converted: this returns `None` for them, so a
    /// caller that wants either shape asks for both.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F32(v) => Some(f64::from(*v)),
            Self::F64(v) => Some(*v),
            Self::F32Array(v) if v.len() == 1 => Some(f64::from(v[0])),
            Self::F64Array(v) if v.len() == 1 => Some(v[0]),
            _ => None,
        }
    }

    /// Every float the value holds, with a scalar reading as one element.
    ///
    /// Returns `None` for a non-float value.
    pub fn to_f64s(&self) -> Option<Vec<f64>> {
        match self {
            Self::F32(v) => Some(vec![f64::from(*v)]),
            Self::F32Array(v) => Some(v.iter().copied().map(f64::from).collect()),
            Self::F64(v) => Some(vec![*v]),
            Self::F64Array(v) => Some(v.clone()),
            _ => None,
        }
    }

    /// The name of the type this value holds, such as `f64` or `ascii_string[]`.
    ///
    /// This enum is `#[non_exhaustive]`, so a caller that reaches its own `_`
    /// arm cannot name what it received. This names every value, including a
    /// variant added later.
    ///
    /// ```
    /// use hdf5_pure::AttrValue;
    ///
    /// assert_eq!(AttrValue::F64(1.5).type_name(), "f64");
    /// assert_eq!(AttrValue::AsciiStringArray(vec![]).type_name(), "ascii_string[]");
    /// ```
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::F32(_) => "f32",
            Self::F32Array(_) => "f32[]",
            Self::F64(_) => "f64",
            Self::F64Array(_) => "f64[]",
            Self::I8(_) => "i8",
            Self::I8Array(_) => "i8[]",
            Self::I16(_) => "i16",
            Self::I16Array(_) => "i16[]",
            Self::I32(_) => "i32",
            Self::I32Array(_) => "i32[]",
            Self::I64(_) => "i64",
            Self::I64Array(_) => "i64[]",
            Self::U8(_) => "u8",
            Self::U8Array(_) => "u8[]",
            Self::U16(_) => "u16",
            Self::U16Array(_) => "u16[]",
            Self::U32(_) => "u32",
            Self::U32Array(_) => "u32[]",
            Self::U64(_) => "u64",
            Self::U64Array(_) => "u64[]",
            Self::String(_) => "string",
            Self::StringSized { .. } => "sized_string",
            Self::StringArray(_) => "string[]",
            Self::StringArraySized { .. } => "sized_string[]",
            Self::AsciiString(_) => "ascii_string",
            Self::AsciiStringSized { .. } => "sized_ascii_string",
            Self::AsciiStringArray(_) => "ascii_string[]",
            Self::AsciiStringArraySized { .. } => "sized_ascii_string[]",
            Self::VarLenAsciiCharArray(_) => "vlen_ascii_char[]",
            Self::VarLenString(_) => "vlen_string",
            Self::VarLenStringArray(_) => "vlen_string[]",
            Self::VarLenAsciiString(_) => "vlen_ascii_string",
            Self::VarLenAsciiStringArray(_) => "vlen_ascii_string[]",
        }
    }
}

/// Every element of an integer array as `T`, for
/// [`AttrValue::int_elements`]'s array arms. One element that `T` cannot hold
/// refuses the whole array, since a partial answer would be indistinguishable
/// from a complete one.
fn many_ints<T: TryFrom<i128>, E: Into<i128> + Copy>(values: &[E]) -> Option<Vec<T>> {
    values.iter().map(|&e| T::try_from(e.into()).ok()).collect()
}

/// How many array elements [`AttrValue`] writes before eliding the rest.
///
/// An attribute array can hold thousands of elements, and a message quoting one
/// has to stay readable. A matter of taste.
const ATTR_DISPLAY_MAX_ELEMENTS: usize = 8;

impl fmt::Display for AttrValue {
    /// The value, not its type: `1.5`, `"metres"`, `[1, 2, 3]`.
    ///
    /// Every element goes through `Debug`, which quotes a string and keeps the
    /// point on a float, so `1.0` does not read as an integer. Long arrays are
    /// elided; use `Debug` for the whole value.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32(v) => write!(f, "{v:?}"),
            Self::F64(v) => write!(f, "{v:?}"),
            Self::I8(v) => write!(f, "{v}"),
            Self::I16(v) => write!(f, "{v}"),
            Self::I32(v) => write!(f, "{v}"),
            Self::I64(v) => write!(f, "{v}"),
            Self::U8(v) => write!(f, "{v}"),
            Self::U16(v) => write!(f, "{v}"),
            Self::U32(v) => write!(f, "{v}"),
            Self::U64(v) => write!(f, "{v}"),
            Self::String(v)
            | Self::AsciiString(v)
            | Self::StringSized { value: v, .. }
            | Self::AsciiStringSized { value: v, .. }
            | Self::VarLenString(v)
            | Self::VarLenAsciiString(v) => write!(f, "{v:?}"),
            Self::F32Array(v) => write_elements(f, v),
            Self::F64Array(v) => write_elements(f, v),
            Self::I8Array(v) => write_elements(f, v),
            Self::I16Array(v) => write_elements(f, v),
            Self::I32Array(v) => write_elements(f, v),
            Self::I64Array(v) => write_elements(f, v),
            Self::U8Array(v) => write_elements(f, v),
            Self::U16Array(v) => write_elements(f, v),
            Self::U32Array(v) => write_elements(f, v),
            Self::U64Array(v) => write_elements(f, v),
            Self::StringArray(v)
            | Self::AsciiStringArray(v)
            | Self::VarLenAsciiCharArray(v)
            | Self::VarLenStringArray(v)
            | Self::VarLenAsciiStringArray(v)
            | Self::StringArraySized { values: v, .. }
            | Self::AsciiStringArraySized { values: v, .. } => write_elements(f, v),
        }
    }
}

/// A bracketed element list, elided past [`ATTR_DISPLAY_MAX_ELEMENTS`].
fn write_elements<T: fmt::Debug>(f: &mut fmt::Formatter<'_>, values: &[T]) -> fmt::Result {
    f.write_str("[")?;
    for (i, value) in values.iter().take(ATTR_DISPLAY_MAX_ELEMENTS).enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{value:?}")?;
    }
    write_elided(f, values.len().saturating_sub(ATTR_DISPLAY_MAX_ELEMENTS))?;
    f.write_str("]")
}

// ---- Dataset builder ----

/// Configuration for SHINES provenance metadata.
#[cfg(feature = "provenance")]
#[derive(Debug, Clone)]
pub struct ProvenanceConfig {
    pub creator: String,
    pub timestamp: String,
    pub source: Option<String>,
}

/// Everything [`DatasetBuilder::with_raw_chunks_lazy`] needs to re-emit a chunked
/// dataset by copying its source chunks verbatim (no decode/re-encode): the
/// per-chunk sizes/masks (enough to plan the destination layout without reading
/// any bytes), a provider that yields each chunk's bytes on demand at write time,
/// the source filter-pipeline message, and the chunk geometry. Built by repack
/// from a source [`Dataset`](crate::Dataset).
pub(crate) struct RawChunkPayload {
    /// Logical chunk dimensions (rank entries, not the trailing element size).
    pub(crate) chunk_dims: Vec<u64>,
    /// Datatype element size in bytes, proven non-zero.
    pub(crate) element_size: NonZeroUsize,
    /// The verbatim source `FilterPipeline` message bytes, if the source had one.
    pub(crate) pipeline_message: Option<Vec<u8>>,
    /// Per-chunk sizes + filter masks in dense row-major grid order, one per slot.
    pub(crate) meta: Vec<ChunkMeta>,
    /// Yields each chunk's compressed bytes on demand during the write, so no
    /// more than one chunk's bytes are resident. Owns its source (e.g. an
    /// `Arc<File>`), so it carries no borrowed lifetime.
    ///
    /// Wrapped in [`AssertUnwindSafe`](core::panic::AssertUnwindSafe) so the
    /// boxed trait object does not strip the `UnwindSafe`/`RefUnwindSafe`
    /// auto-traits from the public builder types that transitively hold it
    /// (removing an auto-trait impl is a semver break). The assertion is sound:
    /// the provider performs only immutable reads and leaves no broken state on
    /// a panic. `ChunkProvider: Send + Sync` keeps the other two auto-traits.
    pub(crate) provider: core::panic::AssertUnwindSafe<Box<dyn ChunkProvider>>,
}

/// Everything [`DatasetBuilder::with_produced_data`] needs to emit a contiguous
/// dataset whose element bytes are produced at write time: the region's total
/// size (known from geometry, which is why the layout never has to read it), the
/// block size the producer is called with, and the producer itself.
///
/// The region is a plain run of bytes, so it is laid out exactly as if the bytes
/// had been handed over — a produced dataset and a materialized one are the same
/// file.
pub(crate) struct ProducedPayload {
    /// Bytes the whole data region occupies.
    pub(crate) total_bytes: u64,
    /// Bytes per block. The final block carries whatever remains.
    pub(crate) block_bytes: u64,
    /// Yields each block's bytes on demand during the write. Wrapped in
    /// [`AssertUnwindSafe`](core::panic::AssertUnwindSafe) for the same reason
    /// [`RawChunkPayload::provider`] is, and soundly so for the same reason.
    pub(crate) provider: core::panic::AssertUnwindSafe<Box<dyn ChunkProvider>>,
}

/// One element of an object-reference dataset written through the builder.
///
/// A reference either names an object by path (resolved to that object's
/// destination address during serialization) or carries a raw address verbatim.
/// The raw form preserves a null reference (address 0) or an undefined reference
/// (`HADDR_UNDEF`, all-ones) exactly, which a faithful rewrite (repack) needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ObjectRefTarget {
    /// Resolve to the destination address of the object at this path.
    Path(String),
    /// Write this exact 8-byte address (e.g. 0 for null, `u64::MAX` for
    /// undefined).
    Raw(u64),
}

/// One object-reference address to resolve during serialization, and where in
/// the dataset's element bytes it sits.
///
/// The offset is explicit rather than implied by a fixed stride so that a
/// reference embedded in a larger element — a compound member, or an array entry
/// — is rewritten in place alongside the fixed-size bytes around it. A dataset
/// whose datatype *is* an object reference simply has one patch every 8 bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ObjectRefPatch {
    /// Byte offset of the 8-byte address within the dataset's element bytes.
    pub byte_offset: usize,
    /// What to write there.
    pub target: ObjectRefTarget,
}

/// Write a resolved object-reference address into a dataset's element bytes.
///
/// The offset comes from the datatype's own layout and `raw` is sized from the
/// same datatype, so a slot that does not fit means the two disagree — a bug in
/// whoever staged them, not input this can correct. The debug assertion catches
/// that in the test suite; in release the write is skipped rather than allowed to
/// corrupt a neighbouring field. Note what a skip leaves behind depends on the
/// caller: placeholder zeros (a null reference) for a dataset staged by
/// [`DatasetBuilder::with_object_references`], but the *source's* address for one
/// staged by [`DatasetBuilder::with_embedded_object_references`], whose buffer is
/// the source's element bytes. Neither is reachable without the assertion firing
/// first.
pub(crate) fn write_reference_address(raw: &mut [u8], byte_offset: usize, address: u64) {
    debug_assert!(
        byte_offset + 8 <= raw.len(),
        "object-reference slot at {byte_offset} does not fit {} element bytes",
        raw.len()
    );
    if let Some(slot) = raw.get_mut(byte_offset..byte_offset + 8) {
        slot.copy_from_slice(&address.to_le_bytes());
    }
}

/// Builder for datasets.
///
/// # Element types
///
/// `DatasetBuilder` exposes a typed method per supported element type. Each sets both the dataset's
/// data and its HDF5 datatype, and defaults the shape to the 1-D `[len]` unless
/// [`with_shape`](Self::with_shape) has already set one. All integer and float types are written
/// little-endian.
///
/// | Method | HDF5 type |
/// |---|---|
/// | [`with_data`](Self::with_data) (generic over [`H5Element`](crate::H5Element)) | Inferred from the element type |
/// | [`with_f64_data`](Self::with_f64_data) | IEEE 64-bit float |
/// | [`with_f32_data`](Self::with_f32_data) | IEEE 32-bit float |
/// | [`with_i8_data`](Self::with_i8_data) / [`with_i16_data`](Self::with_i16_data) / [`with_i32_data`](Self::with_i32_data) / [`with_i64_data`](Self::with_i64_data) | Signed integers |
/// | [`with_u8_data`](Self::with_u8_data) / [`with_u16_data`](Self::with_u16_data) / [`with_u32_data`](Self::with_u32_data) / [`with_u64_data`](Self::with_u64_data) | Unsigned integers |
/// | [`with_ascii_strings`](Self::with_ascii_strings) / [`with_strings`](Self::with_strings) (and their `_sized` forms) | Fixed-width strings, null-padded, ASCII or UTF-8 |
/// | [`with_vlen_strings`](Self::with_vlen_strings) | Variable-length UTF-8 strings (global heap) |
/// | [`with_complex32_data`](Self::with_complex32_data) | Compound `{real: f32, imag: f32}` |
/// | [`with_complex64_data`](Self::with_complex64_data) | Compound `{real: f64, imag: f64}` |
/// | [`with_compound_data`](Self::with_compound_data) | Arbitrary compound types (explicit datatype + raw bytes) |
/// | [`with_compound_values`](Self::with_compound_values) | Safely encoded numeric tuples (field by field) |
/// | [`with_enum_i32_data`](Self::with_enum_i32_data) / [`with_enum_u8_data`](Self::with_enum_u8_data) | Enumeration types |
/// | [`with_array_data`](Self::with_array_data) | Fixed-size array types |
/// | [`with_path_references`](Self::with_path_references) | Object references (resolved by path) |
/// | [`with_dtype`](Self::with_dtype) + [`with_shape`](Self::with_shape) | Empty / zero-dimension datasets |
///
/// The generic [`with_data(&[T])`](Self::with_data) is the counterpart of the typed family, and it
/// dispatches to the matching `with_*_data` method on `T`. See the
/// [generic I/O guide](crate::_guide::generic_io).
///
/// ```rust
/// use hdf5_pure::FileBuilder;
///
/// let mut builder = FileBuilder::new();
/// builder.create_dataset("temperature")
///     .with_f64_data(&[22.5, 23.1, 21.8])
///     .with_shape(&[3]);
/// # let file = hdf5_pure::File::from_bytes(builder.finish()?)?;
/// # assert_eq!(file.dataset("temperature")?.read_f64()?, vec![22.5, 23.1, 21.8]);
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
///
/// [`with_complex32_data`](Self::with_complex32_data) and
/// [`with_complex64_data`](Self::with_complex64_data) accept `&[(f32, f32)]` / `&[(f64, f64)]` and
/// store each pair as a two-field compound. [`with_compound_data`](Self::with_compound_data) takes
/// an explicit [`Datatype`], the raw element bytes, and the element count, so the caller is
/// responsible for the bytes matching the datatype's on-disk layout.
/// [`with_array_data`](Self::with_array_data) takes the array's base [`Datatype`], the array
/// dimensions, the raw element bytes, and the element count, and builds the [`Datatype::Array`]
/// itself. [`with_compound_values`](Self::with_compound_values) is the safe alternative for numeric
/// tuples: it encodes each field explicitly through the [`CompoundType`] trait without copying Rust
/// tuple padding. [`with_enum_i32_data`](Self::with_enum_i32_data) /
/// [`with_enum_u8_data`](Self::with_enum_u8_data) take a datatype (built with [`EnumTypeBuilder`])
/// plus the raw values.
///
/// One lower-level method sits under those,
/// [`with_raw_data(datatype, raw_data, num_elements)`](Self::with_raw_data), which writes an
/// explicit datatype and its raw bytes verbatim, and
/// [`with_compound_data`](Self::with_compound_data) delegates to it.
pub struct DatasetBuilder {
    pub(crate) name: String,
    pub(crate) datatype: Option<Datatype>,
    pub(crate) shape: Option<Vec<u64>>,
    pub(crate) maxshape: Option<Vec<u64>>,
    pub(crate) data: Option<Vec<u8>>,
    pub(crate) attrs: Vec<(String, AttrSpec)>,
    pub(crate) chunk_options: ChunkOptions,
    /// When set, this dataset's chunks are copied verbatim from a source file
    /// (repack's verbatim path): the already-compressed chunk bytes, the source
    /// filter-pipeline message, and the geometry needed to lay them out. This
    /// takes precedence over `data` / `chunk_options` for chunked storage.
    pub(crate) raw_chunks: Option<RawChunkPayload>,
    /// When set, this dataset is contiguous and its element bytes are produced
    /// at write time rather than staged in `data`, which stays `None`.
    pub(crate) produced: Option<ProducedPayload>,
    /// When set, this dataset is an object-reference dataset whose element
    /// addresses are resolved (per-element by path, or written raw) during file
    /// serialization once every object's destination address is known.
    pub(crate) reference_targets: Option<Vec<ObjectRefPatch>>,
    /// When set, this dataset stores variable-length strings: `data` holds the
    /// 16-byte references with placeholder heap addresses, and this staging
    /// carries the global heap collection plus the mask of references to patch
    /// once the post-data cursor is known.
    pub(crate) vl_string_staging: Option<VlStringStaging>,
    /// A user-defined fill value, encoded in the dataset's datatype (little-
    /// endian, one element wide). `None` leaves the crate's library-default fill
    /// value message untouched. Its byte width is checked against the datatype's
    /// element size when the dataset is serialized.
    pub(crate) fill: Option<Vec<u8>>,
    /// Whether this dataset allocates storage at all. `Unallocated` declares the
    /// shape, datatype and fill value and writes no data region, which is what
    /// preserves a never-written dataset through a rewrite rather than
    /// materializing a grid of fill values (issue #293).
    pub(crate) allocation: StorageAllocation,
    /// Where this dataset's element type is written: in its own header, or as a
    /// reference to a committed datatype object named by path.
    pub(crate) datatype_location: DatatypeLocation,
    #[cfg(feature = "provenance")]
    pub(crate) provenance: Option<ProvenanceConfig>,
}

impl DatasetBuilder {
    pub(crate) fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            datatype: None,
            shape: None,
            maxshape: None,
            data: None,
            attrs: Vec::new(),
            chunk_options: ChunkOptions::default(),
            raw_chunks: None,
            produced: None,
            reference_targets: None,
            vl_string_staging: None,
            fill: None,
            allocation: StorageAllocation::Allocated,
            datatype_location: DatatypeLocation::Inline,
            #[cfg(feature = "provenance")]
            provenance: None,
        }
    }

    /// Write this dataset's element type as a reference to the committed
    /// datatype at `path` instead of encoding it in the dataset's own header.
    ///
    /// The dataset still declares its element type the usual way (through
    /// `with_*_data` or [`with_dtype`](Self::with_dtype)); this says where that
    /// type is *stored*. The two must agree — a dataset naming a committed type
    /// it does not match is refused when the file is written, because every
    /// reader would believe the committed one and read the element bytes wrong.
    ///
    /// `path` names a datatype committed with
    /// [`FileBuilder::commit_datatype`](crate::FileBuilder::commit_datatype) or
    /// [`GroupBuilder::commit_datatype`], with or without a leading `/`.
    pub fn with_committed_datatype(&mut self, path: &str) -> &mut Self {
        self.datatype_location = DatatypeLocation::CommittedPath(normalize_object_path(path));
        self
    }

    /// Attach an attribute whose datatype is the committed one at `path`.
    ///
    /// The same agreement rule as [`with_committed_datatype`](Self::with_committed_datatype)
    /// applies: `value` decides the attribute's type, and it must be the type
    /// committed at `path`.
    pub fn set_attr_committed(&mut self, name: &str, value: AttrValue, path: &str) -> &mut Self {
        self.attrs
            .push((name.to_string(), committed_attr_spec(name, &value, path)));
        self
    }

    pub fn with_f64_data(&mut self, data: &[f64]) -> &mut Self {
        self.datatype = Some(make_f64_type());
        let mut b = Vec::with_capacity(data.len() * 8);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.set_element_bytes(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_f32_data(&mut self, data: &[f32]) -> &mut Self {
        self.datatype = Some(make_f32_type());
        let mut b = Vec::with_capacity(data.len() * 4);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.set_element_bytes(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_i32_data(&mut self, data: &[i32]) -> &mut Self {
        self.datatype = Some(make_i32_type());
        let mut b = Vec::with_capacity(data.len() * 4);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.set_element_bytes(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_i64_data(&mut self, data: &[i64]) -> &mut Self {
        self.datatype = Some(make_i64_type());
        let mut b = Vec::with_capacity(data.len() * 8);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.set_element_bytes(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_u8_data(&mut self, data: &[u8]) -> &mut Self {
        self.datatype = Some(make_u8_type());
        self.set_element_bytes(data.to_vec());
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_i8_data(&mut self, data: &[i8]) -> &mut Self {
        self.datatype = Some(make_i8_type());
        let mut b = Vec::with_capacity(data.len());
        for &v in data {
            b.push(v as u8);
        }
        self.set_element_bytes(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_i16_data(&mut self, data: &[i16]) -> &mut Self {
        self.datatype = Some(make_i16_type());
        let mut b = Vec::with_capacity(data.len() * 2);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.set_element_bytes(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_u16_data(&mut self, data: &[u16]) -> &mut Self {
        self.datatype = Some(make_u16_type());
        let mut b = Vec::with_capacity(data.len() * 2);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.set_element_bytes(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_u32_data(&mut self, data: &[u32]) -> &mut Self {
        self.datatype = Some(make_u32_type());
        let mut b = Vec::with_capacity(data.len() * 4);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.set_element_bytes(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    pub fn with_u64_data(&mut self, data: &[u64]) -> &mut Self {
        self.datatype = Some(make_u64_type());
        let mut b = Vec::with_capacity(data.len() * 8);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.set_element_bytes(b);
        if self.shape.is_none() {
            self.shape = Some(vec![data.len() as u64]);
        }
        self
    }

    /// Write an object reference dataset. Each address is an 8-byte object-header
    /// address *relative to the file's base address*, which is the form HDF5
    /// stores and the form a reference read back out of a file carries. The two
    /// coincide on a file with no userblock; on one with a userblock — every
    /// MATLAB v7.3 `.mat` — an absolute offset here would be wrong by the
    /// userblock's size.
    ///
    /// The addresses are written as given — nothing here resolves them, and
    /// nothing checks that they name anything. Use
    /// [`with_path_references`](Self::with_path_references) to name targets by
    /// path and have the writer resolve them. In a
    /// [`File::open_rw`](crate::File::open_rw) session the addresses are checked
    /// at `commit` against the objects that commit deletes and the headers it
    /// rewrites elsewhere, so an address the commit is about to vacate is
    /// refused rather than written, with a message naming
    /// [`with_path_references`](Self::with_path_references) as the alternative.
    pub fn with_reference_data(&mut self, addresses: &[u64]) -> &mut Self {
        self.datatype = Some(make_object_reference_type());
        let mut b = Vec::with_capacity(addresses.len() * 8);
        for &addr in addresses {
            b.extend_from_slice(&addr.to_le_bytes());
        }
        self.set_element_bytes(b);
        if self.shape.is_none() {
            self.shape = Some(vec![addresses.len() as u64]);
        }
        self
    }

    /// Write an object reference dataset by path. During file serialization,
    /// each path is resolved to the absolute address of the named object.
    /// Paths use `/` separators (e.g., `"#refs#/child1"`).
    pub fn with_path_references(&mut self, paths: &[&str]) -> &mut Self {
        let targets = paths
            .iter()
            .map(|s| ObjectRefTarget::Path(s.to_string()))
            .collect();
        self.with_object_references(targets)
    }

    /// Write an object-reference dataset from explicit per-element targets,
    /// preserving null/undefined references verbatim. The datatype is set to the
    /// 8-byte object-reference type; each [`ObjectRefTarget::Path`] is resolved to
    /// its destination address during serialization, while
    /// [`ObjectRefTarget::Raw`] is written as-is. This is the faithful re-emit
    /// path used by repack. The shape defaults to `[targets.len()]` unless
    /// [`with_shape`](Self::with_shape) sets it.
    pub(crate) fn with_object_references(&mut self, targets: Vec<ObjectRefTarget>) -> &mut Self {
        self.datatype = Some(make_object_reference_type());
        // Placeholder zeros — patched once all destination addresses are known.
        self.set_element_bytes(vec![0u8; targets.len() * 8]);
        if self.shape.is_none() {
            self.shape = Some(vec![targets.len() as u64]);
        }
        self.reference_targets = Some(
            targets
                .into_iter()
                .enumerate()
                .map(|(i, target)| ObjectRefPatch {
                    byte_offset: i * 8,
                    target,
                })
                .collect(),
        );
        self
    }

    /// Write a dataset whose datatype *contains* object references without being
    /// one — a compound with a reference member, or an array of them.
    ///
    /// `raw` is the source's element bytes; each [`ObjectRefPatch`] names an
    /// address within them to resolve during serialization. Every other byte is
    /// carried through untouched, so the fixed-size members keep their exact
    /// stored bytes. The shape defaults to `[num_elements]` unless
    /// [`with_shape`](Self::with_shape) sets it.
    pub(crate) fn with_embedded_object_references(
        &mut self,
        datatype: Datatype,
        raw: Vec<u8>,
        num_elements: u64,
        patches: Vec<ObjectRefPatch>,
    ) -> &mut Self {
        self.datatype = Some(datatype);
        self.set_element_bytes(raw);
        if self.shape.is_none() {
            self.shape = Some(vec![num_elements]);
        }
        self.reference_targets = Some(patches);
        self
    }

    /// Write a complex32 (f32 real/imag pair) dataset.
    pub fn with_complex32_data(&mut self, data: &[(f32, f32)]) -> &mut Self {
        self.with_complex_data(data)
    }

    /// Write a complex64 (f64 real/imag pair) dataset.
    pub fn with_complex64_data(&mut self, data: &[(f64, f64)]) -> &mut Self {
        self.with_complex_data(data)
    }

    /// Write a complex dataset of any component width: a two-field `{real,
    /// imag}` compound with `real` at offset 0 and `imag` at
    /// `size_of::<T>()`, which is the layout MATLAB v7.3 uses for a complex
    /// array of the component's class.
    pub(crate) fn with_complex_data<T: ComplexComponent>(&mut self, data: &[(T, T)]) -> &mut Self {
        let ct = CompoundTypeBuilder::new()
            .field("real", T::datatype())
            .field("imag", T::datatype())
            .build()
            .expect("two fields of a nonzero-width component");
        let width = size_of::<T>();
        let mut raw = vec![0u8; data.len() * 2 * width];
        for (slot, &(re, im)) in raw.chunks_exact_mut(2 * width).zip(data) {
            let (real, imag) = slot.split_at_mut(width);
            re.encode_le_into(real);
            im.encode_le_into(imag);
        }
        self.with_compound_data(ct, raw, data.len() as u64)
    }

    /// Write a compound (struct) dataset.
    pub fn with_compound_data(
        &mut self,
        datatype: Datatype,
        raw_data: Vec<u8>,
        num_elements: u64,
    ) -> &mut Self {
        self.with_raw_data(datatype, raw_data, num_elements)
    }

    /// Write a dataset from an explicit datatype and its raw element bytes.
    ///
    /// The lowest-level data entry point: `raw_data` is written verbatim as the
    /// dataset's storage, interpreted by `datatype`, so the caller is
    /// responsible for the bytes matching the datatype's on-disk layout (little
    /// endian, `num_elements` elements each of the datatype's size). It underpins
    /// the typed helpers and lets a captured `(datatype, bytes)` pair — for
    /// example from reading an existing dataset — be re-emitted without a typed
    /// helper. The shape defaults to `[num_elements]` unless
    /// [`with_shape`](Self::with_shape) sets it.
    pub fn with_raw_data(
        &mut self,
        datatype: Datatype,
        raw_data: Vec<u8>,
        num_elements: u64,
    ) -> &mut Self {
        self.datatype = Some(datatype);
        self.set_element_bytes(raw_data);
        if self.shape.is_none() {
            self.shape = Some(vec![num_elements]);
        }
        self
    }

    /// Replace the staged element bytes, dropping any staging that described the
    /// *previous* ones.
    ///
    /// [`vl_string_staging`](Self::vl_string_staging) and
    /// [`reference_targets`](Self::reference_targets) are both descriptions of
    /// what is in `data`: which element references still hold a placeholder, and
    /// at which byte offsets. Replacing the bytes invalidates both, so every
    /// entry point that sets element data goes through here rather than
    /// assigning the field — the invariant is "the staging describes `data`",
    /// and it is one a new setter would otherwise have to know to uphold.
    ///
    /// Leaving one behind was reachable and silent: `with_raw_data` after
    /// `with_vlen_strings` kept a staging that owned one patch offset while
    /// `data` held a whole new array, so the commit patched element 0 and wrote
    /// the caller's own bytes into the rest. Where those bytes were element
    /// references read out of another dataset, two datasets ended up naming one
    /// global heap collection with nothing recording it (issue #321). A shorter
    /// replacement was worse than silent: `patch_vl_refs_masked` indexes at the
    /// staged offsets unguarded, so it panicked out of `commit`.
    ///
    /// The four setters that *do* establish a staging assign it immediately
    /// after their call to this, which is why clearing here does not defeat
    /// them.
    fn set_element_bytes(&mut self, data: Vec<u8>) {
        self.data = Some(data);
        self.vl_string_staging = None;
        self.reference_targets = None;
    }

    /// Stage a dataset that declares its shape and element type and allocates no
    /// storage.
    ///
    /// By default the reference library does not allocate a *contiguous or
    /// chunked* dataset's storage until something is written to it — compact
    /// data is inline in the layout message and is always present — so one
    /// created and never written holds nothing: no index structure under a
    /// chunked layout, an undefined data address under a contiguous one. Reading it answers the
    /// fill value for every element (issue #292), which is why a
    /// rewrite cannot recover this state from the values it reads back — the
    /// dataset that stores a grid of fill values reads identically. Repack
    /// carries it across by saying so here instead (issue #293).
    ///
    /// The chunk geometry, maximum shape, filters and fill value are set the
    /// usual way and are all reproduced; only the data region is absent. Unlike
    /// every other data entry point this stages no bytes, so the shape/data
    /// agreement the writer enforces does not apply to it.
    pub(crate) fn with_unallocated_storage(
        &mut self,
        datatype: Datatype,
        dims: &[u64],
    ) -> &mut Self {
        self.datatype = Some(datatype);
        if self.shape.is_none() {
            self.shape = Some(dims.to_vec());
        }
        self.allocation = StorageAllocation::Unallocated;
        self
    }

    /// Stage a chunked dataset whose chunks are streamed verbatim from a source
    /// file one at a time, without decoding or re-encoding any chunk and without
    /// holding more than one chunk's bytes in memory.
    ///
    /// Repack's out-of-core verbatim path: `meta` is the per-chunk sizes + filter
    /// masks in dense row-major chunk-grid order (one per slot), and `provider`
    /// yields each chunk's already-compressed bytes on demand at write time. The
    /// destination layout is computed from `meta` alone, so the chunks are never
    /// all resident at once. `pipeline_message` is the source's `FilterPipeline`
    /// message bytes, reused as-is so every filter — including ones this crate
    /// cannot itself apply (ZFP, SZIP, unknown) — is reproduced byte-for-byte.
    /// `dims`/`maxshape`/`chunk_dims`/`element_size` describe the geometry. The
    /// shape defaults to `dims` and the chunk dimensions to `chunk_dims`. The
    /// provider owns its source (e.g. an `Arc<File>`), so this carries no
    /// borrowed lifetime.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_raw_chunks_lazy(
        &mut self,
        datatype: Datatype,
        dims: &[u64],
        maxshape: Option<&[u64]>,
        chunk_dims: &[u64],
        element_size: NonZeroUsize,
        pipeline_message: Option<Vec<u8>>,
        meta: Vec<ChunkMeta>,
        provider: Box<dyn ChunkProvider>,
    ) -> &mut Self {
        // The whole-chunk byte size that drives the fixed/extensible-array
        // chunk-size encoding width is not carried here: `chunk_dims` and
        // `element_size` are, and the width is derived from them where it is
        // used (`chunked_write::full_chunk_bytes`), so this payload cannot
        // disagree with the geometry it travels beside.
        self.datatype = Some(datatype);
        if self.shape.is_none() {
            self.shape = Some(dims.to_vec());
        }
        if let Some(ms) = maxshape {
            self.maxshape = Some(ms.to_vec());
        }
        self.chunk_options.chunk_dims = Some(chunk_dims.to_vec());
        self.raw_chunks = Some(RawChunkPayload {
            chunk_dims: chunk_dims.to_vec(),
            element_size,
            pipeline_message,
            meta,
            provider: core::panic::AssertUnwindSafe(provider),
        });
        self
    }

    /// Stage a contiguous dataset whose element bytes are produced at write time,
    /// one block at a time, rather than handed over as a slice.
    ///
    /// The data region is `total_bytes` long — pure geometry, so the layout pass
    /// never touches the producer — and the emitter pulls `block_bytes` at a time
    /// (the last block being whatever remains). The result is byte-for-byte the
    /// dataset the same content staged through [`with_raw_data`](Self::with_raw_data)
    /// would produce; only the peak memory differs.
    ///
    /// The caller owns the contract that `total_bytes` matches `shape` and the
    /// datatype's element size; a block of the wrong length is refused at write
    /// time rather than written.
    pub(crate) fn with_produced_data(
        &mut self,
        datatype: Datatype,
        shape: &[u64],
        total_bytes: u64,
        block_bytes: u64,
        provider: Box<dyn ChunkProvider>,
    ) -> &mut Self {
        debug_assert!(block_bytes > 0, "a block must make progress");
        // A produced region is contiguous and its bytes never exist in `data`, so
        // it cannot take part in anything that patches or re-encodes them. Each of
        // these would fail differently and quietly — chunking would encode the
        // empty `data` and never call the producer, and the two patch passes would
        // write into a buffer that is not the dataset. Callers construct these
        // one way, so this pins the invariant rather than validating input.
        debug_assert!(
            self.vl_string_staging.is_none()
                && self.reference_targets.is_none()
                && self.maxshape.is_none()
                && !self.chunk_options.is_chunked(),
            "a produced dataset is plain contiguous storage: no VL staging, \
             references, maxshape, or chunking"
        );
        self.datatype = Some(datatype);
        if self.shape.is_none() {
            self.shape = Some(shape.to_vec());
        }
        self.produced = Some(ProducedPayload {
            total_bytes,
            block_bytes,
            provider: core::panic::AssertUnwindSafe(provider),
        });
        self
    }

    /// Safely encode a slice of compound values field by field.
    ///
    /// Built-in implementations support numeric tuples with one through twelve
    /// fields. No Rust struct or tuple padding is copied into the file.
    pub fn with_compound_values<T: CompoundType>(
        &mut self,
        values: &[T],
    ) -> Result<&mut Self, crate::error::FormatError> {
        let datatype = T::datatype()?;
        if !matches!(datatype, Datatype::Compound { .. }) {
            return Err(crate::error::FormatError::TypeMismatch {
                expected: "Compound",
                actual: "non-Compound",
            });
        }
        let element_size = datatype.type_size().to_usize()?;
        if element_size == 0 {
            return Err(crate::error::FormatError::InvalidCompoundSize);
        }
        let mut raw = Vec::with_capacity(values.len().saturating_mul(element_size));
        for value in values {
            let start = raw.len();
            value.encode(&mut raw);
            let actual = raw.len() - start;
            if actual != element_size {
                return Err(crate::error::FormatError::DataSizeMismatch {
                    expected: element_size,
                    actual,
                });
            }
        }
        Ok(self.with_compound_data(datatype, raw, values.len() as u64))
    }

    /// Write an enum dataset with i32 values.
    pub fn with_enum_i32_data(&mut self, datatype: Datatype, values: &[i32]) -> &mut Self {
        self.datatype = Some(datatype);
        let mut raw = Vec::with_capacity(values.len() * 4);
        for &v in values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        self.set_element_bytes(raw);
        if self.shape.is_none() {
            self.shape = Some(vec![values.len() as u64]);
        }
        self
    }

    /// Write an enum dataset with u8 values.
    pub fn with_enum_u8_data(&mut self, datatype: Datatype, values: &[u8]) -> &mut Self {
        self.datatype = Some(datatype);
        self.set_element_bytes(values.to_vec());
        if self.shape.is_none() {
            self.shape = Some(vec![values.len() as u64]);
        }
        self
    }

    /// Write a variable-length UTF-8 string dataset.
    ///
    /// Each element is stored as a global-heap object holding the string's
    /// bytes; the datatype is `H5T_VLEN { H5T_STRING { STRSIZE=VAR, ASCII or
    /// UTF-8 } }`. The shape defaults to `[values.len()]` unless
    /// [`with_shape`](Self::with_shape) sets it (use that for ND VL strings,
    /// passing `values` in row-major order).
    ///
    /// Empty strings become zero-length heap objects (reading back as `""`).
    /// This convenience method cannot distinguish a null element from an empty
    /// one, nor carry embedded NULs, non-UTF-8 payloads, or a specific
    /// charset/padding.
    pub fn with_vlen_strings(&mut self, values: &[&str]) -> &mut Self {
        // Staged straight from the caller's strings: owning each one first would
        // copy the whole payload for nothing (see [`stage_vl_payloads`]). A VL
        // string's base type is one byte, so the reference length is the byte
        // count.
        self.stage_vlen(
            make_vlen_string_type(CharacterSet::Utf8),
            values.len() as u64,
            stage_vl_payloads(values.iter().map(|s| Some(s.as_bytes())), NonZeroUsize::MIN),
        );
        self
    }

    /// Write a fixed-width ASCII string dataset, sized to the longest value.
    ///
    /// Every element occupies the same number of bytes — the length of the
    /// longest value, or one byte when they are all empty — zero-padded on the
    /// right, and the datatype is `H5T_STRING { STRSIZE = width, NULLPAD,
    /// ASCII }`. This is the encoding
    /// [`AttrValue::AsciiStringArray`] writes as an attribute, so the same
    /// values land on disk the same way whichever they are stored as, and
    /// [`Dataset::read_string`](crate::Dataset::read_string) reads them back
    /// without the caller trimming anything.
    ///
    /// The shape defaults to `[values.len()]` unless
    /// [`with_shape`](Self::with_shape) sets it. Use
    /// [`with_ascii_strings_sized`](Self::with_ascii_strings_sized) to declare a
    /// width instead of deriving one, [`with_strings`](Self::with_strings) for
    /// UTF-8, and [`with_vlen_strings`](Self::with_vlen_strings) when elements
    /// should not share a width at all.
    ///
    /// Only a value with no fixed-width HDF5 datatype at all — one longer than
    /// the datatype message's 4-byte size field can describe — is refused, with
    /// [`FormatError::FixedStringTooLong`].
    ///
    /// ```
    /// use hdf5_pure::{File, FileBuilder};
    ///
    /// let mut fb = FileBuilder::new();
    /// fb.create_dataset("station")
    ///     .with_ascii_strings(&["north", "s", "east"])
    ///     .unwrap();
    /// let file = File::from_bytes(fb.finish().unwrap()).unwrap();
    /// let ds = file.dataset("station").unwrap();
    /// assert_eq!(ds.read_string().unwrap(), ["north", "s", "east"]);
    /// ```
    pub fn with_ascii_strings(&mut self, values: &[&str]) -> Result<&mut Self, FormatError> {
        self.stage_fixed_strings(values, derived_string_width(values), CharacterSet::Ascii)
    }

    /// Write a fixed-width ASCII string dataset of a width you declare.
    ///
    /// The width-deriving [`with_ascii_strings`](Self::with_ascii_strings) sizes
    /// the datatype to the values in hand, which is the wrong rule when the
    /// values in hand are not all the values: a dataset can be extended, and a
    /// later, longer string would have nowhere to go. Declaring the width is
    /// also how a rewrite reproduces a source dataset's own `STRSIZE`, and how a
    /// writer matches a column width that came from a schema rather than from
    /// data.
    ///
    /// A value longer than `width` is refused with
    /// [`FormatError::FixedStringTooLong`] rather than stored as a prefix, and a
    /// `width` of zero with [`FormatError::ZeroFixedStringWidth`], since no HDF5
    /// string datatype may be zero bytes wide. Shorter values are zero-padded,
    /// as `NULLPAD` declares.
    ///
    /// ```
    /// use hdf5_pure::{File, FileBuilder, FormatError};
    ///
    /// let mut fb = FileBuilder::new();
    /// fb.create_dataset("station")
    ///     .with_ascii_strings_sized(&["north", "s"], 16)
    ///     .unwrap();
    /// let file = File::from_bytes(fb.finish().unwrap()).unwrap();
    /// assert_eq!(
    ///     file.dataset("station").unwrap().read_string().unwrap(),
    ///     ["north", "s"]
    /// );
    ///
    /// // A value the declared width cannot hold is refused, not truncated.
    /// let mut fb = FileBuilder::new();
    /// assert!(matches!(
    ///     fb.create_dataset("station").with_ascii_strings_sized(&["north"], 2),
    ///     Err(FormatError::FixedStringTooLong { index: 0, .. })
    /// ));
    /// ```
    pub fn with_ascii_strings_sized(
        &mut self,
        values: &[&str],
        width: u32,
    ) -> Result<&mut Self, FormatError> {
        let width = checked_width(values, width)?;
        self.stage_fixed_strings(values, width, CharacterSet::Ascii)
    }

    /// Write a fixed-width UTF-8 string dataset, sized to the longest value.
    ///
    /// The UTF-8 counterpart of
    /// [`with_ascii_strings`](Self::with_ascii_strings), standing to it as
    /// [`AttrValue::StringArray`] stands to [`AttrValue::AsciiStringArray`]: the
    /// same encoding under a different charset bit. The width is a count of
    /// *bytes*, so a value with multi-byte characters takes more of it than its
    /// character count suggests.
    pub fn with_strings(&mut self, values: &[&str]) -> Result<&mut Self, FormatError> {
        self.stage_fixed_strings(values, derived_string_width(values), CharacterSet::Utf8)
    }

    /// Write a fixed-width UTF-8 string dataset of a width you declare.
    ///
    /// The UTF-8 counterpart of
    /// [`with_ascii_strings_sized`](Self::with_ascii_strings_sized), refusing
    /// the same two ways. `width` counts bytes rather than characters, and a
    /// value is measured whole: this never splits one across the width boundary,
    /// because it never truncates at all.
    pub fn with_strings_sized(
        &mut self,
        values: &[&str],
        width: u32,
    ) -> Result<&mut Self, FormatError> {
        let width = checked_width(values, width)?;
        self.stage_fixed_strings(values, width, CharacterSet::Utf8)
    }

    /// Shared body of the fixed-width string entry points.
    ///
    /// The declared `STRSIZE` and the padding applied to every element come from
    /// the one `width` here, which is what keeps the datatype message and the
    /// element bytes describing the same thing. Other paddings (`NULLTERM`,
    /// `SPACEPAD`) and other charsets stay the business of
    /// [`with_raw_data`](Self::with_raw_data) with a hand-built
    /// [`Datatype::String`].
    fn stage_fixed_strings(
        &mut self,
        values: &[&str],
        width: NonZeroU32,
        charset: CharacterSet,
    ) -> Result<&mut Self, FormatError> {
        let raw = encode_fixed_strings(values, width)?;
        self.datatype = Some(Datatype::String {
            size: width.get(),
            padding: StringPadding::NullPad,
            charset,
        });
        self.set_element_bytes(raw);
        if self.shape.is_none() {
            self.shape = Some(vec![values.len() as u64]);
        }
        Ok(self)
    }

    /// Write a variable-length string dataset from an explicit source datatype
    /// and per-element byte payloads, preserving the null-vs-empty distinction.
    ///
    /// `datatype` must be a string-shaped variable-length datatype
    /// (`is_string: true`, or the MATLAB `H5T_VLEN { H5T_STRING { STRSIZE=1 } }`
    /// shape); its charset, padding, and base type are reproduced verbatim. Each
    /// [`VlStringElement`] is either a null reference or a heap object holding
    /// exact bytes. This is the faithful re-emit path used by repack. Returns a
    /// [`TypeMismatch`](crate::FormatError::TypeMismatch) if `datatype` is not a
    /// VL-string datatype. The shape defaults to `[elements.len()]` unless
    /// [`with_shape`](Self::with_shape) sets it.
    pub(crate) fn with_vlen_string_elements(
        &mut self,
        datatype: Datatype,
        elements: &[VlStringElement],
    ) -> Result<&mut Self, crate::error::FormatError> {
        if !crate::vl_data::is_vlen_string_datatype(&datatype) {
            return Err(crate::error::FormatError::TypeMismatch {
                expected: "VariableLength string",
                actual: "non-VariableLength string",
            });
        }
        self.stage_vlen_strings(datatype, elements);
        Ok(self)
    }

    /// Shared body of the VL-string write entry points: stage the references
    /// and global heap collection and record them on the builder.
    fn stage_vlen_strings(&mut self, datatype: Datatype, elements: &[VlStringElement]) {
        // A VL string's base type is one byte, so the reference length is the
        // byte count (element_size = 1).
        self.stage_vlen_elements(datatype, elements, NonZeroUsize::MIN);
    }

    /// Write a *non-string* variable-length (sequence) dataset from an explicit
    /// source datatype and per-element byte payloads.
    ///
    /// `datatype` must be a non-string VL datatype (e.g. `H5T_VLEN
    /// { H5T_NATIVE_DOUBLE }`); its base type is reproduced verbatim. Each
    /// element's exact heap bytes are re-staged through a fresh global heap, and
    /// the per-element reference stores the base-type element count. This is the
    /// faithful re-emit path used by repack. Returns a
    /// [`TypeMismatch`](crate::FormatError::TypeMismatch) if `datatype` is a
    /// string-shaped VL datatype or not variable-length at all. The shape
    /// defaults to `[elements.len()]` unless [`with_shape`](Self::with_shape)
    /// sets it.
    pub(crate) fn with_vlen_sequence_elements(
        &mut self,
        datatype: Datatype,
        elements: &[VlStringElement],
    ) -> Result<&mut Self, crate::error::FormatError> {
        let Datatype::VariableLength { base_type, .. } = &datatype else {
            return Err(crate::error::FormatError::TypeMismatch {
                expected: "non-string VariableLength",
                actual: "non-VariableLength",
            });
        };
        if crate::vl_data::is_vlen_string_datatype(&datatype) {
            return Err(crate::error::FormatError::TypeMismatch {
                expected: "non-string VariableLength",
                actual: "VariableLength string",
            });
        }
        let Some(element_size) = NonZeroUsize::new(base_type.type_size() as usize) else {
            return Err(crate::error::FormatError::VlDataError(
                "non-string VL base type has zero size".into(),
            ));
        };
        self.stage_vlen_elements(datatype, elements, element_size);
        Ok(self)
    }

    /// Write a dataset whose datatype *contains* variable-length members without
    /// being variable-length itself — a compound with a VL member, or an array of
    /// such compounds.
    ///
    /// `raw` is the source's element bytes and `offsets` gives the byte offset of
    /// every embedded VL reference within them, paired in order with `elements`
    /// (each either a null reference or the exact heap bytes it named). The
    /// references are re-stamped to point at this file's own global heap, so the
    /// source's addresses never survive into the output. This is the faithful
    /// re-emit path used by repack; the shape defaults to `[num_elements]` unless
    /// [`with_shape`](Self::with_shape) sets it.
    pub(crate) fn with_embedded_vlen_elements(
        &mut self,
        datatype: Datatype,
        raw: Vec<u8>,
        num_elements: u64,
        offsets: &[usize],
        elements: &[VlStringElement],
    ) -> &mut Self {
        let (element_bytes, staging) = stage_embedded_vl_elements(raw, offsets, elements);
        self.datatype = Some(datatype);
        self.set_element_bytes(element_bytes);
        self.vl_string_staging = Some(staging);
        if self.shape.is_none() {
            self.shape = Some(vec![num_elements]);
        }
        self
    }

    /// Shared body of the VL write entry points (string and sequence): stage the
    /// references and global heap collection and record them on the builder.
    fn stage_vlen_elements(
        &mut self,
        datatype: Datatype,
        elements: &[VlStringElement],
        element_size: NonZeroUsize,
    ) {
        let n = elements.len() as u64;
        self.stage_vlen(datatype, n, stage_vl_elements(elements, element_size));
    }

    /// Record staged variable-length element bytes and their heap collections on
    /// the builder.
    fn stage_vlen(
        &mut self,
        datatype: Datatype,
        num_elements: u64,
        (element_bytes, staging): (Vec<u8>, VlStringStaging),
    ) {
        self.datatype = Some(datatype);
        self.set_element_bytes(element_bytes);
        self.vl_string_staging = Some(staging);
        if self.shape.is_none() {
            self.shape = Some(vec![num_elements]);
        }
    }

    /// Write an array-typed dataset.
    pub fn with_array_data(
        &mut self,
        base_type: Datatype,
        array_dims: &[u32],
        raw_data: Vec<u8>,
        num_elements: u64,
    ) -> &mut Self {
        self.datatype = Some(Datatype::Array {
            base_type: Box::new(base_type),
            dimensions: array_dims.to_vec(),
        });
        self.set_element_bytes(raw_data);
        if self.shape.is_none() {
            self.shape = Some(vec![num_elements]);
        }
        self
    }

    /// Declare the dataset's dimensions.
    ///
    /// The shape and the staged element data have to agree on the element
    /// count, or the write is refused with
    /// [`FormatError::ShapeDataMismatch`].
    /// A shape holding a zero dimension declares no elements and is held to that
    /// same rule: stage an empty slice, or no data at all beside a
    /// [`with_dtype`](Self::with_dtype), rather than data with nowhere to go.
    pub fn with_shape(&mut self, shape: &[u64]) -> &mut Self {
        self.shape = Some(shape.to_vec());
        self
    }

    /// Set the datatype without providing data.
    /// Use with `with_shape` for empty/zero-dimension datasets.
    pub fn with_dtype(&mut self, dt: Datatype) -> &mut Self {
        self.datatype = Some(dt);
        self
    }

    /// Set maximum dimensions for a resizable dataset.
    /// Use `u64::MAX` for unlimited dimensions.
    pub fn with_maxshape(&mut self, maxshape: &[u64]) -> &mut Self {
        self.maxshape = Some(maxshape.to_vec());
        self
    }

    pub fn set_attr(&mut self, name: &str, value: AttrValue) -> &mut Self {
        self.attrs.push((name.to_string(), AttrSpec::Value(value)));
        self
    }

    /// Attach an already-encoded attribute message, written exactly as given.
    ///
    /// See [`AttrSpec::Verbatim`] for what this preserves that `set_attr` cannot,
    /// and for the datatypes it must not be used with.
    pub(crate) fn set_attr_verbatim(&mut self, message: AttributeMessage) -> &mut Self {
        self.attrs
            .push((message.name.clone(), AttrSpec::Verbatim(message)));
        self
    }

    /// Attach a variable-length string attribute with the given datatype and
    /// dataspace, staging `strings` into a heap of this file's own.
    /// See [`AttrSpec::VerbatimVarLen`].
    pub(crate) fn set_attr_var_len_verbatim(
        &mut self,
        mut message: AttributeMessage,
        strings: Vec<String>,
    ) -> &mut Self {
        message.raw_data = vl_string_reference_bytes(&strings);
        self.attrs.push((
            message.name.clone(),
            AttrSpec::VerbatimVarLen { message, strings },
        ));
        self
    }

    /// Enable chunked storage with given chunk dimensions.
    pub fn with_chunks(&mut self, chunk_dims: &[u64]) -> &mut Self {
        self.chunk_options.chunk_dims = Some(chunk_dims.to_vec());
        self
    }

    /// Enable deflate compression (implies chunked if not already set).
    ///
    /// Mutually exclusive with [`with_lzf`](Self::with_lzf), which fills the
    /// same byte-compressor slot, and with [`with_zfp`](Self::with_zfp), which
    /// replaces it: requesting either combination makes the write fail with a
    /// filter error. May follow [`with_shuffle`](Self::with_shuffle) or
    /// [`with_scale_offset`](Self::with_scale_offset).
    pub fn with_deflate(&mut self, level: u32) -> &mut Self {
        self.chunk_options.set_filter(FilterKind::Deflate(level));
        self
    }

    /// Enable shuffle filter (usually combined with deflate or LZF).
    ///
    /// Mutually exclusive with the two filters that consume the raw elements
    /// themselves, [`with_scale_offset`](Self::with_scale_offset) and
    /// [`with_zfp`](Self::with_zfp): requesting shuffle alongside either makes
    /// the write fail with a filter error rather than dropping it.
    pub fn with_shuffle(&mut self) -> &mut Self {
        self.chunk_options.set_filter(FilterKind::Shuffle);
        self
    }

    /// Enable LZF compression (implies chunked if not already set).
    ///
    /// LZF (h5py filter id 32000) is a fast, lossless byte compressor with a
    /// lower compression ratio than deflate. h5py reads and writes it out of
    /// the box; the plain C library needs h5py's filter plugin. Usually
    /// combined with [`with_shuffle`](Self::with_shuffle); mutually exclusive
    /// with [`with_deflate`](Self::with_deflate), which fills the same
    /// byte-compressor slot, and with [`with_zfp`](Self::with_zfp), which
    /// replaces it — requesting either combination makes the write fail with a
    /// filter error.
    pub fn with_lzf(&mut self) -> &mut Self {
        self.chunk_options.set_filter(FilterKind::Lzf);
        self
    }

    /// Enable fletcher32 checksum.
    pub fn with_fletcher32(&mut self) -> &mut Self {
        self.chunk_options.set_filter(FilterKind::Fletcher32);
        self
    }

    /// Enable scale-offset compression (implies chunked if not already set).
    ///
    /// Scale-offset stores each chunk's values as offsets from the chunk
    /// minimum, packed into the fewest bits the chunk's range needs:
    ///
    /// * [`ScaleOffset::Integer`] is **lossless** for integer datasets. Pass
    ///   `0` to let the encoder choose the bit width per chunk (the usual
    ///   choice).
    /// * [`ScaleOffset::FloatDScale`] is **lossy** for float datasets: values
    ///   are rounded to the given number of decimal digits before packing.
    ///
    /// The datatype class/sign/byte-order are derived from the dataset's
    /// datatype when the file is written, so the mode must match the data
    /// (integer mode on `with_i*`/`with_u*` data, float mode on
    /// `with_f32`/`with_f64` data) or `finish()` / `write()` returns a
    /// [`FormatError`]. Scale-offset consumes the raw
    /// elements itself, so it is mutually exclusive with
    /// [`with_zfp`](Self::with_zfp) and [`with_shuffle`](Self::with_shuffle) —
    /// requesting either alongside it makes the write fail with a filter error
    /// — but it may be followed by [`with_deflate`](Self::with_deflate) or
    /// [`with_lzf`](Self::with_lzf). Files are readable by the reference HDF5
    /// library (filter id 6) and vice versa.
    pub fn with_scale_offset(&mut self, mode: ScaleOffset) -> &mut Self {
        self.chunk_options
            // `Defined` named rather than defaulted: it is this builder that
            // chooses it on the caller's behalf, matching what the reference
            // library records for every dataset whose fill value is not
            // explicitly undefined, and `with_scale_offset` is where a reader
            // looks for that choice.
            .set_filter(FilterKind::ScaleOffset(mode, FillAvailability::Defined));
        self
    }

    /// Enable ZFP fixed-rate compression (implies chunked if not already set).
    ///
    /// `rate` is the number of compressed bits per value. Supports f32, f64,
    /// i32, and i64 datasets in 1D–4D. ZFP is a standalone compressor that
    /// consumes the raw elements itself, so it is mutually exclusive with
    /// [`with_shuffle`](Self::with_shuffle),
    /// [`with_scale_offset`](Self::with_scale_offset),
    /// [`with_deflate`](Self::with_deflate) and [`with_lzf`](Self::with_lzf):
    /// requesting any of them alongside it makes the write fail with a filter
    /// error.
    ///
    /// The scalar type is derived from the dataset's datatype when the file
    /// is written, so any of `with_{f32,f64,i32,i64}_data` or an explicit
    /// `with_dtype` establishes it. `finish()` / `write()` returns
    /// [`FormatError::UnsupportedZfp`] if the dataset's datatype isn't one of
    /// the four supported scalar types, or if the chunk rank is outside 1..=4.
    ///
    /// The resulting file is byte-compatible with the reference H5Z-ZFP
    /// plugin (HDF5 filter ID 32013): other tools like h5py + hdf5plugin
    /// will read and decompress it, and vice versa.
    #[cfg(feature = "zfp")]
    pub fn with_zfp(&mut self, rate: f64) -> &mut Self {
        self.chunk_options.set_filter(FilterKind::Zfp(rate));
        self
    }

    /// Attach SHINES provenance metadata (SHA-256, creator, timestamp).
    ///
    /// The SHA-256 hash of the raw dataset bytes is computed automatically
    /// during file serialization and stored as `_provenance_sha256`.
    #[cfg(feature = "provenance")]
    pub fn with_provenance(
        &mut self,
        creator: &str,
        timestamp: &str,
        source: Option<&str>,
    ) -> &mut Self {
        self.provenance = Some(ProvenanceConfig {
            creator: creator.to_string(),
            timestamp: timestamp.to_string(),
            source: source.map(|s| s.to_string()),
        });
        self
    }
}

// ---- Group builder ----

/// Builder for HDF5 groups.
///
/// Datasets, sub-groups, and attributes can be added in any order before
/// calling [`finish()`](GroupBuilder::finish). This is useful when the full
/// set of attributes is not known up front — for example, building a
/// MATLAB struct where `MATLAB_fields` lists every child dataset name:
///
/// ```rust
/// # use hdf5_pure::{FileBuilder, AttrValue};
/// let mut builder = FileBuilder::new();
/// let mut grp = builder.create_group("my_struct");
///
/// let mut fields = Vec::new();
/// for name in &["x", "y", "z"] {
///     fields.push(name.to_string());
///     grp.create_dataset(name).with_f64_data(&[0.0]);
/// }
///
/// // Attribute set after all children are created
/// grp.set_attr("MATLAB_fields", AttrValue::VarLenAsciiCharArray(fields));
/// builder.add_group(grp.finish());
/// ```
pub struct GroupBuilder {
    pub(crate) name: String,
    pub(crate) datasets: Vec<DatasetBuilder>,
    pub(crate) sub_groups: Vec<FinishedGroup>,
    pub(crate) attrs: Vec<(String, AttrSpec)>,
    pub(crate) committed: Vec<CommittedDatatype>,
}

impl GroupBuilder {
    pub(crate) fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            datasets: Vec::new(),
            sub_groups: Vec::new(),
            attrs: Vec::new(),
            committed: Vec::new(),
        }
    }

    pub fn create_dataset(&mut self, name: &str) -> &mut DatasetBuilder {
        self.datasets.push(DatasetBuilder::new(name));
        self.datasets.last_mut().unwrap()
    }

    /// Create a nested group builder. Call `.finish()` on it and then
    /// `add_group()` to add it to this group.
    pub fn create_group(&mut self, name: &str) -> GroupBuilder {
        GroupBuilder::new(name)
    }

    /// Add a finished sub-group to this group.
    pub fn add_group(&mut self, group: FinishedGroup) {
        self.sub_groups.push(group);
    }

    pub fn set_attr(&mut self, name: &str, value: AttrValue) {
        self.attrs.push((name.to_string(), AttrSpec::Value(value)));
    }

    /// Attach an already-encoded attribute message, written exactly as given.
    ///
    /// See [`AttrSpec::Verbatim`] for what this preserves that `set_attr` cannot,
    /// and for the datatypes it must not be used with.
    pub(crate) fn set_attr_verbatim(&mut self, message: AttributeMessage) {
        self.attrs
            .push((message.name.clone(), AttrSpec::Verbatim(message)));
    }

    /// Attach an attribute whose datatype is the committed one at `path`.
    ///
    /// See [`DatasetBuilder::set_attr_committed`].
    pub fn set_attr_committed(&mut self, name: &str, value: AttrValue, path: &str) {
        self.attrs
            .push((name.to_string(), committed_attr_spec(name, &value, path)));
    }

    /// Attach a variable-length string attribute with the given datatype and
    /// dataspace, staging `strings` into a heap of this file's own.
    /// See [`AttrSpec::VerbatimVarLen`].
    pub(crate) fn set_attr_var_len_verbatim(
        &mut self,
        mut message: AttributeMessage,
        strings: Vec<String>,
    ) {
        message.raw_data = vl_string_reference_bytes(&strings);
        self.attrs.push((
            message.name.clone(),
            AttrSpec::VerbatimVarLen { message, strings },
        ));
    }

    /// Commit `datatype` in this group under `name`, the way `H5Tcommit` does.
    ///
    /// See [`FileBuilder::commit_datatype`](crate::FileBuilder::commit_datatype)
    /// for what a committed datatype is and how datasets and attributes name one.
    /// The path they use is this group's path joined with `name`.
    pub fn commit_datatype(&mut self, name: &str, datatype: Datatype) {
        self.committed.push(CommittedDatatype {
            name: name.to_string(),
            datatype,
        });
    }

    /// Consume the builder, returning a FinishedGroup to add to FileWriter.
    pub fn finish(self) -> FinishedGroup {
        FinishedGroup {
            name: self.name,
            datasets: self.datasets,
            sub_groups: self.sub_groups,
            attrs: self.attrs,
            committed: self.committed,
        }
    }
}

/// A finished group ready for the file writer.
pub struct FinishedGroup {
    pub(crate) name: String,
    pub(crate) datasets: Vec<DatasetBuilder>,
    pub(crate) sub_groups: Vec<FinishedGroup>,
    pub(crate) attrs: Vec<(String, AttrSpec)>,
    pub(crate) committed: Vec<CommittedDatatype>,
}

/// A datatype to be written as its own named object, which datasets and
/// attributes then reference instead of encoding the type again.
pub(crate) struct CommittedDatatype {
    /// Link name within the owning group.
    pub(crate) name: String,
    pub(crate) datatype: Datatype,
}

/// Canonicalize a path naming an object in the file being written.
///
/// The writer's own path map is keyed without a leading slash (the root group is
/// the empty path), while an HDF5 user writes `/mytype`. Both forms name the same
/// object, so both are accepted and reduced to the map's form here rather than at
/// every lookup.
pub(crate) fn normalize_object_path(path: &str) -> String {
    path.trim_matches('/').to_string()
}

/// How an attribute named after the committed datatype at `path` is written:
/// an already-encoded message whose datatype has moved out of it.
///
/// The value still decides the message's dataspace and raw bytes; only the
/// datatype moves out. The writer checks the two agree before it lays anything
/// out.
///
/// A value that keeps its data in the global heap takes
/// [`VerbatimVarLen`](AttrSpec::VerbatimVarLen) rather than
/// [`Verbatim`](AttrSpec::Verbatim), so the writer stages its collections and
/// patches the placeholder addresses [`build_attr_message`] wrote. Plain
/// `Verbatim` is for bytes that are already final, and carrying a
/// variable-length value through it wrote every element pointing at address 0 —
/// a file that returned `Ok`, dropped the attribute from `attrs`, and read as
/// an empty string in the C library.
pub(crate) fn committed_attr_spec(name: &str, value: &AttrValue, path: &str) -> AttrSpec {
    let mut message = build_attr_message(name, value);
    message.datatype_location = DatatypeLocation::CommittedPath(normalize_object_path(path));
    match value.var_len_strings() {
        Some(strings) => AttrSpec::VerbatimVarLen {
            message,
            strings: strings.to_vec(),
        },
        None => AttrSpec::Verbatim(message),
    }
}

#[cfg(test)]
mod attr_value_accessor_tests {
    use super::AttrValue;

    /// Every representation of a single string reaches `as_str`. The point of
    /// the accessor is that a caller need not know which one a file yielded.
    #[test]
    fn as_str_spans_every_single_string_shape() {
        for value in [
            AttrValue::String("double".into()),
            AttrValue::AsciiString("double".into()),
            AttrValue::StringArray(vec!["double".into()]),
            AttrValue::AsciiStringArray(vec!["double".into()]),
            AttrValue::VarLenAsciiCharArray(vec!["double".into()]),
            AttrValue::string_sized("double", 32).unwrap(),
            AttrValue::ascii_string_sized("double", 32).unwrap(),
            AttrValue::string_array_sized(vec!["double".into()], 32).unwrap(),
            AttrValue::ascii_string_array_sized(vec!["double".into()], 32).unwrap(),
        ] {
            assert_eq!(value.as_str(), Some("double"), "{value:?}");
        }
    }

    /// "Exactly one" is the contract: an array of two is not a single string,
    /// and neither is a numeric value.
    #[test]
    fn as_str_rejects_non_single_strings() {
        for value in [
            AttrValue::StringArray(vec!["a".into(), "b".into()]),
            AttrValue::AsciiStringArray(vec!["a".into(), "b".into()]),
            AttrValue::VarLenAsciiCharArray(vec![]),
            AttrValue::string_array_sized(vec!["a".into(), "b".into()], 8).unwrap(),
            AttrValue::ascii_string_array_sized(vec![], 8).unwrap(),
            AttrValue::F64(1.5),
            AttrValue::I64(3),
        ] {
            assert_eq!(value.as_str(), None, "{value:?}");
        }
    }

    /// A scalar is viewed as a one-element slice, and the view borrows: no
    /// element is copied to produce it.
    #[test]
    fn as_strings_reads_a_scalar_as_one_element() {
        for value in [
            AttrValue::String("m/s".into()),
            AttrValue::AsciiString("m/s".into()),
            AttrValue::string_sized("m/s", 8).unwrap(),
            AttrValue::ascii_string_sized("m/s", 8).unwrap(),
        ] {
            let seen = value.as_strings().expect("a string value");
            assert_eq!(seen, ["m/s"], "{value:?}");
            assert_eq!(seen.len(), 1, "{value:?}");
        }
    }

    #[test]
    fn as_strings_keeps_every_element_of_each_array_shape() {
        let fields: Vec<String> = vec!["x".into(), "y".into(), "velocity".into()];
        for value in [
            AttrValue::StringArray(fields.clone()),
            AttrValue::AsciiStringArray(fields.clone()),
            AttrValue::VarLenAsciiCharArray(fields.clone()),
            AttrValue::string_array_sized(fields.clone(), 16).unwrap(),
            AttrValue::ascii_string_array_sized(fields.clone(), 16).unwrap(),
        ] {
            assert_eq!(
                value.as_strings().expect("a string value"),
                ["x", "y", "velocity"],
                "{value:?}"
            );
        }
    }

    /// An empty string array is still a string array. An empty slice and `None`
    /// mean different things — no elements against not a string at all — and a
    /// caller distinguishing them depends on this.
    #[test]
    fn as_strings_separates_empty_from_absent() {
        let empty = AttrValue::StringArray(vec![]);
        assert_eq!(empty.as_strings(), Some(&[][..]));
        assert_eq!(AttrValue::I64(1).as_strings(), None);
    }

    /// One value per integer variant, scalar and one-element array alike, for
    /// the accessors that must span every width. Signed variants carry `-7`
    /// where they can, so a sign lost in the widening shows up as a value.
    fn every_integer_variant() -> Vec<(AttrValue, i64)> {
        vec![
            (AttrValue::I8(-7), -7),
            (AttrValue::I16(-7), -7),
            (AttrValue::I32(-7), -7),
            (AttrValue::I64(-7), -7),
            (AttrValue::U8(7), 7),
            (AttrValue::U16(7), 7),
            (AttrValue::U32(7), 7),
            (AttrValue::U64(7), 7),
            (AttrValue::I8Array(vec![-7]), -7),
            (AttrValue::I16Array(vec![-7]), -7),
            (AttrValue::I32Array(vec![-7]), -7),
            (AttrValue::I64Array(vec![-7]), -7),
            (AttrValue::U8Array(vec![7]), 7),
            (AttrValue::U16Array(vec![7]), 7),
            (AttrValue::U32Array(vec![7]), 7),
            (AttrValue::U64Array(vec![7]), 7),
        ]
    }

    /// Every width reads as `i64`, which is what lets a caller ignore the width
    /// the file happened to use — the point of keeping it on the variant (#350)
    /// is that it is *available*, not that it must be handled.
    #[test]
    fn as_i64_widens_every_integer_variant() {
        for (value, expected) in every_integer_variant() {
            assert_eq!(value.as_i64(), Some(expected), "{}", value.type_name());
            assert_eq!(
                value.to_i64s(),
                Some(vec![expected]),
                "{} through the plural accessor",
                value.type_name()
            );
        }
    }

    /// The same span for `u64`, where the signed variants' `-7` has no value and
    /// the unsigned ones do.
    #[test]
    fn as_u64_widens_every_integer_variant() {
        for (value, expected) in every_integer_variant() {
            let expected = u64::try_from(expected).ok();
            assert_eq!(value.as_u64(), expected, "{}", value.type_name());
            assert_eq!(
                value.to_u64s(),
                expected.map(|v| vec![v]),
                "{} through the plural accessor",
                value.type_name()
            );
        }
    }

    /// The full range of each unsigned width reads as itself rather than
    /// wrapping through a narrower conversion on the way to `u64`.
    #[test]
    fn the_widest_value_of_each_width_reads_as_itself() {
        assert_eq!(AttrValue::U8(u8::MAX).as_u64(), Some(255));
        assert_eq!(AttrValue::U16(u16::MAX).as_u64(), Some(65_535));
        assert_eq!(AttrValue::U32(u32::MAX).as_u64(), Some(4_294_967_295));
        assert_eq!(AttrValue::I8(i8::MIN).as_i64(), Some(-128));
        assert_eq!(AttrValue::I16(i16::MIN).as_i64(), Some(-32_768));
        assert_eq!(AttrValue::I32(i32::MIN).as_i64(), Some(-2_147_483_648));
        assert_eq!(
            AttrValue::U8Array(vec![u8::MAX, 0]).to_i64s(),
            Some(vec![255, 0])
        );
        assert_eq!(
            AttrValue::I16Array(vec![i16::MIN, i16::MAX]).to_i64s(),
            Some(vec![-32_768, 32_767])
        );
    }

    /// A `u64` past `i64::MAX` has no `i64` value, and a negative number has no
    /// `u64`. Reporting `None` is the contract; a wrapping cast either way would
    /// hand back a plausible wrong number instead.
    #[test]
    fn scalar_accessors_refuse_a_value_that_does_not_fit() {
        let past_max = (i64::MAX as u64) + 1;
        assert_eq!(AttrValue::U64(u64::MAX).as_i64(), None);
        assert_eq!(AttrValue::U64(past_max).as_i64(), None);
        assert_eq!(AttrValue::U64Array(vec![past_max]).as_i64(), None);
        assert_eq!(
            AttrValue::U64(i64::MAX as u64).as_i64(),
            Some(i64::MAX),
            "the largest value that does fit must still be readable"
        );

        assert_eq!(AttrValue::I64(-1).as_u64(), None);
        assert_eq!(AttrValue::I32(-1).as_u64(), None);
        assert_eq!(AttrValue::I64Array(vec![-1]).as_u64(), None);
        assert_eq!(AttrValue::I64(0).as_u64(), Some(0));
    }

    #[test]
    fn as_i64_rejects_multi_element_and_non_integer() {
        assert_eq!(AttrValue::I64Array(vec![1, 2]).as_i64(), None);
        assert_eq!(AttrValue::U64Array(vec![1, 2]).as_i64(), None);
        assert_eq!(AttrValue::F64(1.0).as_i64(), None);
        assert_eq!(AttrValue::String("1".into()).as_i64(), None);
        assert_eq!(AttrValue::F64(1.0).as_u64(), None);
    }

    #[test]
    fn to_i64s_reads_scalars_and_arrays_alike() {
        assert_eq!(AttrValue::I64(4).to_i64s(), Some(vec![4]));
        assert_eq!(AttrValue::I32(4).to_i64s(), Some(vec![4]));
        assert_eq!(AttrValue::U32(4).to_i64s(), Some(vec![4]));
        // A `u64` attribute is what a C-written `H5T_NATIVE_UINT64` arrives as,
        // so this arm carries real traffic.
        assert_eq!(AttrValue::U64(4).to_i64s(), Some(vec![4]));
        assert_eq!(
            AttrValue::I64Array(vec![1, 2, 3]).to_i64s(),
            Some(vec![1, 2, 3])
        );
        assert_eq!(AttrValue::U64Array(vec![1, 2]).to_i64s(), Some(vec![1, 2]));
        assert_eq!(AttrValue::I64Array(vec![]).to_i64s(), Some(vec![]));
        assert_eq!(AttrValue::U64Array(vec![]).to_i64s(), Some(vec![]));
        assert_eq!(AttrValue::F64Array(vec![1.0]).to_i64s(), None);
    }

    #[test]
    fn to_u64s_reads_scalars_and_arrays_alike() {
        assert_eq!(AttrValue::U64(4).to_u64s(), Some(vec![4]));
        assert_eq!(AttrValue::U32(4).to_u64s(), Some(vec![4]));
        assert_eq!(AttrValue::I64(4).to_u64s(), Some(vec![4]));
        assert_eq!(
            AttrValue::U64Array(vec![1, u64::MAX]).to_u64s(),
            Some(vec![1, u64::MAX])
        );
        assert_eq!(AttrValue::I64Array(vec![1, 2]).to_u64s(), Some(vec![1, 2]));
        assert_eq!(AttrValue::F64(1.0).to_u64s(), None);
    }

    /// The range rule is per element, not per variant. A single out-of-range
    /// element rejects the whole read rather than wrapping that one silently —
    /// the guarantee `as_i64` documents has to hold at every length, or it is
    /// the shape-dependent behavior these accessors exist to remove.
    #[test]
    fn plural_accessors_apply_the_range_rule_to_every_element() {
        let past_max = (i64::MAX as u64) + 1;
        assert_eq!(AttrValue::U64Array(vec![1, past_max]).to_i64s(), None);
        assert_eq!(AttrValue::U64Array(vec![past_max, 1]).to_i64s(), None);
        assert_eq!(AttrValue::U64(u64::MAX).to_i64s(), None);
        assert_eq!(
            AttrValue::U64Array(vec![1, i64::MAX as u64]).to_i64s(),
            Some(vec![1, i64::MAX]),
            "every element fitting must still read"
        );

        assert_eq!(AttrValue::I64Array(vec![1, -1]).to_u64s(), None);
        assert_eq!(AttrValue::I64Array(vec![-1, 1]).to_u64s(), None);
        assert_eq!(AttrValue::I64(-1).to_u64s(), None);
    }

    #[test]
    fn as_f64_reads_one_float_from_either_shape() {
        assert_eq!(AttrValue::F64(1.5).as_f64(), Some(1.5));
        assert_eq!(AttrValue::F64Array(vec![1.5]).as_f64(), Some(1.5));
        assert_eq!(AttrValue::F64Array(vec![1.5, 2.5]).as_f64(), None);
    }

    /// The float accessors span both widths, so a caller that wants the number
    /// need not know whether the file stored four bytes or eight (#354).
    /// Widening an `f32` is exact, which the extremes of its range state.
    #[test]
    fn the_float_accessors_span_both_widths() {
        assert_eq!(AttrValue::F32(1.5).as_f64(), Some(1.5));
        assert_eq!(AttrValue::F32Array(vec![1.5]).as_f64(), Some(1.5));
        assert_eq!(AttrValue::F32Array(vec![1.5, 2.5]).as_f64(), None);
        assert_eq!(AttrValue::F32(f32::MAX).as_f64(), Some(f64::from(f32::MAX)));
        assert_eq!(AttrValue::F32(1.5).to_f64s(), Some(vec![1.5]));
        assert_eq!(
            AttrValue::F32Array(vec![f32::MIN, f32::MAX]).to_f64s(),
            Some(vec![f64::from(f32::MIN), f64::from(f32::MAX)])
        );
        assert_eq!(AttrValue::F32Array(vec![]).to_f64s(), Some(vec![]));
        // An integer is still not a float, at either width.
        assert_eq!(AttrValue::F32(1.0).as_i64(), None);
        assert_eq!(AttrValue::I32(1).as_f64(), None);
    }

    /// The float accessors do not convert integers. A caller that accepts
    /// either asks for both, rather than having a silent widening decided here.
    #[test]
    fn float_accessors_do_not_convert_integers() {
        assert_eq!(AttrValue::I64(1).as_f64(), None);
        assert_eq!(AttrValue::U32(1).as_f64(), None);
        assert_eq!(AttrValue::I64Array(vec![1]).to_f64s(), None);
        assert_eq!(AttrValue::U64Array(vec![1]).to_f64s(), None);
    }

    #[test]
    fn to_f64s_reads_scalars_and_arrays_alike() {
        assert_eq!(AttrValue::F64(1.5).to_f64s(), Some(vec![1.5]));
        assert_eq!(
            AttrValue::F64Array(vec![1.5, 2.5]).to_f64s(),
            Some(vec![1.5, 2.5])
        );
        assert_eq!(AttrValue::F64Array(vec![]).to_f64s(), Some(vec![]));
        assert_eq!(AttrValue::String("1.5".into()).to_f64s(), None);
    }
}

#[cfg(all(test, feature = "std"))]
mod attr_value_display_tests {
    use super::{ATTR_DISPLAY_MAX_ELEMENTS, AttrValue};

    /// One value of every `AttrValue` variant.
    ///
    /// The match below names each variant with no `_` arm, so a variant added to
    /// the enum stops this module compiling until it is named there — which is
    /// the prompt to add it to the list above, the thing the tests actually walk.
    /// Without it, a test that walks "every variant" walks only the ones that
    /// existed when it was written, and #350 added ten at once. The match runs
    /// over the values for want of a way to write it once; its arms are empty
    /// because the check is the compiler's, not the run's.
    fn one_of_every_variant() -> Vec<AttrValue> {
        let values = vec![
            AttrValue::F32(0.0),
            AttrValue::F32Array(vec![]),
            AttrValue::F64(0.0),
            AttrValue::F64Array(vec![]),
            AttrValue::I8(0),
            AttrValue::I8Array(vec![]),
            AttrValue::I16(0),
            AttrValue::I16Array(vec![]),
            AttrValue::I32(0),
            AttrValue::I32Array(vec![]),
            AttrValue::I64(0),
            AttrValue::I64Array(vec![]),
            AttrValue::U8(0),
            AttrValue::U8Array(vec![]),
            AttrValue::U16(0),
            AttrValue::U16Array(vec![]),
            AttrValue::U32(0),
            AttrValue::U32Array(vec![]),
            AttrValue::U64(0),
            AttrValue::U64Array(vec![]),
            AttrValue::String(String::new()),
            AttrValue::string_sized("", 4).unwrap(),
            AttrValue::StringArray(vec![]),
            AttrValue::string_array_sized(vec![], 4).unwrap(),
            AttrValue::AsciiString(String::new()),
            AttrValue::ascii_string_sized("", 4).unwrap(),
            AttrValue::AsciiStringArray(vec![]),
            AttrValue::ascii_string_array_sized(vec![], 4).unwrap(),
            AttrValue::VarLenAsciiCharArray(vec![]),
            AttrValue::VarLenString(String::new()),
            AttrValue::VarLenStringArray(vec![]),
            AttrValue::VarLenAsciiString(String::new()),
            AttrValue::VarLenAsciiStringArray(vec![]),
        ];
        for value in &values {
            match value {
                AttrValue::F32(_) | AttrValue::F32Array(_) => {}
                AttrValue::F64(_) | AttrValue::F64Array(_) => {}
                AttrValue::I8(_) | AttrValue::I8Array(_) => {}
                AttrValue::I16(_) | AttrValue::I16Array(_) => {}
                AttrValue::I32(_) | AttrValue::I32Array(_) => {}
                AttrValue::I64(_) | AttrValue::I64Array(_) => {}
                AttrValue::U8(_) | AttrValue::U8Array(_) => {}
                AttrValue::U16(_) | AttrValue::U16Array(_) => {}
                AttrValue::U32(_) | AttrValue::U32Array(_) => {}
                AttrValue::U64(_) | AttrValue::U64Array(_) => {}
                AttrValue::String(_) | AttrValue::StringArray(_) => {}
                AttrValue::StringSized { .. } | AttrValue::StringArraySized { .. } => {}
                AttrValue::AsciiString(_) | AttrValue::AsciiStringArray(_) => {}
                AttrValue::AsciiStringSized { .. } | AttrValue::AsciiStringArraySized { .. } => {}
                AttrValue::VarLenAsciiCharArray(_) => {}
                AttrValue::VarLenString(_) | AttrValue::VarLenStringArray(_) => {}
                AttrValue::VarLenAsciiString(_) | AttrValue::VarLenAsciiStringArray(_) => {}
            }
        }
        values
    }

    /// A caller that fell through its own `_` arm has only this name to report,
    /// so no two variants may share one.
    #[test]
    fn type_name_is_distinct_for_every_variant() {
        let values = one_of_every_variant();

        let mut names: Vec<&str> = values.iter().map(AttrValue::type_name).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "two variants share a type name");
        assert!(!names.contains(&""));
    }

    /// Every variant writes something. A `Display` arm added for a new variant
    /// but left empty would show as nothing at all in the message quoting it,
    /// which reads as an attribute with no value rather than a bug here.
    #[test]
    fn display_writes_something_for_every_variant() {
        for value in one_of_every_variant() {
            assert!(
                !value.to_string().is_empty(),
                "{} writes nothing",
                value.type_name()
            );
        }
    }

    #[test]
    fn display_writes_the_value_not_the_variant() {
        assert_eq!(AttrValue::F64(1.5).to_string(), "1.5");
        assert_eq!(AttrValue::F32(1.5).to_string(), "1.5");
        assert_eq!(AttrValue::F32(1.0).to_string(), "1.0");
        assert_eq!(AttrValue::I8(-7).to_string(), "-7");
        assert_eq!(AttrValue::I16(-7).to_string(), "-7");
        assert_eq!(AttrValue::I32(-7).to_string(), "-7");
        assert_eq!(AttrValue::U8(255).to_string(), "255");
        assert_eq!(AttrValue::U16(65_535).to_string(), "65535");
        assert_eq!(AttrValue::U32(7).to_string(), "7");
        assert_eq!(AttrValue::U8Array(vec![1, 2]).to_string(), "[1, 2]");
        assert_eq!(AttrValue::I16Array(vec![-1, 2]).to_string(), "[-1, 2]");
        assert_eq!(AttrValue::U64(u64::MAX).to_string(), "18446744073709551615");
        assert_eq!(AttrValue::String("metres".into()).to_string(), "\"metres\"");
        assert_eq!(
            AttrValue::I64Array(vec![1, 2, 3]).to_string(),
            "[1, 2, 3]",
            "no `I64Array(..)` wrapper, which is what `Debug` is for"
        );
        assert_eq!(
            AttrValue::StringArray(vec!["a".into(), "b".into()]).to_string(),
            "[\"a\", \"b\"]"
        );
        assert_eq!(AttrValue::F64Array(vec![]).to_string(), "[]");
    }

    /// A float keeps its point, scalar and array alike, so a whole number does
    /// not read as an integer.
    #[test]
    fn display_keeps_the_point_on_a_whole_float() {
        assert_eq!(AttrValue::F64(1.0).to_string(), "1.0");
        assert_eq!(
            AttrValue::F64Array(vec![1.0, 2.5]).to_string(),
            "[1.0, 2.5]"
        );
    }

    #[test]
    fn display_elides_a_long_array_and_reports_the_remainder() {
        let values: Vec<i64> = (0..ATTR_DISPLAY_MAX_ELEMENTS as i64 + 5).collect();
        let shown = AttrValue::I64Array(values).to_string();

        assert!(shown.ends_with(", … 5 more]"), "{shown}");
        assert_eq!(shown.matches(", ").count(), ATTR_DISPLAY_MAX_ELEMENTS);
    }

    /// The boundary: exactly the cap is written whole, with no "0 more".
    #[test]
    fn display_does_not_elide_at_exactly_the_cap() {
        let values: Vec<i64> = (0..ATTR_DISPLAY_MAX_ELEMENTS as i64).collect();
        let shown = AttrValue::I64Array(values).to_string();

        assert!(!shown.contains('…'), "{shown}");
        assert!(shown.ends_with("7]"), "{shown}");
    }
}

#[cfg(test)]
mod fixed_string_tests {
    use super::{
        AttrValue, CharacterSet, DatasetBuilder, DataspaceType, Datatype, FormatError,
        StringPadding, build_attr_message, derived_string_width,
    };

    /// One fixed-width string entry point as a value, so a case table can name
    /// all four without repeating the body under each.
    type Stage = fn(&mut DatasetBuilder) -> Result<&mut DatasetBuilder, FormatError>;

    /// The staged datatype and the staged bytes, which every case here reads
    /// together: a width the message declares and a stride the bytes use are two
    /// halves of one rule, and a test that checked only the message would pass on
    /// a builder that padded to something else.
    fn staged(builder: &DatasetBuilder) -> (Datatype, Vec<u8>) {
        (
            builder.datatype.clone().expect("a datatype"),
            builder.data.clone().expect("element bytes"),
        )
    }

    /// The declared width, having asserted it is the stride the bytes actually
    /// use. `count` is how many elements were written.
    fn declared_width(builder: &DatasetBuilder, count: usize) -> u32 {
        let (dt, raw) = staged(builder);
        let Datatype::String { size, .. } = dt else {
            panic!("expected a string datatype, got {dt:?}");
        };
        assert_eq!(
            raw.len(),
            size as usize * count,
            "{count} elements of a {size}-byte type do not account for {} bytes",
            raw.len()
        );
        size
    }

    #[test]
    fn a_derived_width_is_the_longest_value_and_never_zero() {
        assert_eq!(derived_string_width(&["north", "s", "east"]).get(), 5);
        assert_eq!(derived_string_width(&["s", "east", "north"]).get(), 5);
        // A multi-byte character is measured in bytes, not characters.
        assert_eq!(derived_string_width(&["é", "ab"]).get(), 2);
        // All-empty and no values at all both take the one byte the format's
        // minimum requires, rather than a zero-width datatype.
        assert_eq!(derived_string_width::<&str>(&[]).get(), 1);
        assert_eq!(derived_string_width(&["", ""]).get(), 1);
    }

    #[test]
    fn a_derived_width_dataset_pads_every_element_to_the_longest() {
        let mut b = DatasetBuilder::new("d");
        b.with_ascii_strings(&["north", "s", ""]).unwrap();

        assert_eq!(declared_width(&b, 3), 5);
        let mut expected = Vec::new();
        expected.extend_from_slice(b"north"); // exactly the width, unpadded
        expected.extend_from_slice(b"s\0\0\0\0");
        expected.extend_from_slice(b"\0\0\0\0\0"); // an empty value is all padding
        assert_eq!(
            staged(&b).1,
            expected,
            "each value is zero-padded on the right to the declared width"
        );
        assert_eq!(b.shape, Some(vec![3]));
    }

    #[test]
    fn a_declared_width_is_used_verbatim_even_when_the_values_are_shorter() {
        let mut b = DatasetBuilder::new("d");
        b.with_ascii_strings_sized(&["ab", "c"], 4).unwrap();

        assert_eq!(
            declared_width(&b, 2),
            4,
            "the declared width stands; it is not shrunk to the values in hand"
        );
        assert_eq!(staged(&b).1, b"ab\0\0c\0\0\0".to_vec());
    }

    /// The refusal names *which* value overflowed, so a caller with a thousand
    /// of them can find it. A wrong index is the mutation this catches, which is
    /// why the offending value is neither first nor last.
    #[test]
    fn a_value_longer_than_the_declared_width_is_refused_not_truncated() {
        let mut b = DatasetBuilder::new("d");
        let refused = b.with_ascii_strings_sized(&["ab", "north", "cd"], 4);

        assert!(
            matches!(
                refused,
                Err(FormatError::FixedStringTooLong {
                    index: 1,
                    len: 5,
                    width: 4
                })
            ),
            "{:?}",
            refused.map(|_| ())
        );
        assert!(
            b.datatype.is_none() && b.data.is_none(),
            "a refused call stages nothing"
        );
    }

    /// A width of zero is the one width no HDF5 string datatype may have, and
    /// nothing in the values can make it legal — an empty value fits it, so the
    /// per-value check never fires.
    #[test]
    fn a_declared_width_of_zero_is_refused() {
        for refused in [
            DatasetBuilder::new("d").with_ascii_strings_sized(&[""], 0),
            DatasetBuilder::new("d").with_strings_sized(&[""], 0),
        ] {
            assert!(
                matches!(refused, Err(FormatError::ZeroFixedStringWidth)),
                "{:?}",
                refused.map(|_| ())
            );
        }
    }

    /// The charset bit is the only thing separating the two families, and it is
    /// the one a reader uses to decide how to decode the bytes.
    #[test]
    fn each_entry_point_declares_its_own_charset_and_null_padding() {
        let cases: [(Stage, _); 4] = [
            (|b| b.with_ascii_strings(&["ab"]), CharacterSet::Ascii),
            (
                |b| b.with_ascii_strings_sized(&["ab"], 4),
                CharacterSet::Ascii,
            ),
            (|b| b.with_strings(&["ab"]), CharacterSet::Utf8),
            (|b| b.with_strings_sized(&["ab"], 4), CharacterSet::Utf8),
        ];
        for (stage, expected) in cases {
            let mut b = DatasetBuilder::new("d");
            stage(&mut b).unwrap();
            let Datatype::String {
                padding, charset, ..
            } = staged(&b).0
            else {
                panic!("expected a string datatype");
            };
            assert_eq!(charset, expected);
            assert_eq!(padding, StringPadding::NullPad);
        }
    }

    /// The claim the entry-point docs make: the same values stored as an
    /// attribute and as a dataset land on disk identically, whether the width is
    /// derived or declared. The two sides share [`derived_string_width`] and the
    /// padder under it, so this is what would catch either of them growing a
    /// rule of its own.
    #[test]
    fn a_fixed_string_dataset_matches_the_attribute_encoding_of_the_same_values() {
        const VALUES: [&str; 3] = ["north", "s", ""];
        let owned: Vec<String> = VALUES.iter().map(|s| (*s).to_string()).collect();

        let cases: [(AttrValue, Stage); 4] = [
            (AttrValue::AsciiStringArray(owned.clone()), |b| {
                b.with_ascii_strings(&VALUES)
            }),
            (AttrValue::StringArray(owned.clone()), |b| {
                b.with_strings(&VALUES)
            }),
            (
                AttrValue::ascii_string_array_sized(owned.clone(), 16).unwrap(),
                |b| b.with_ascii_strings_sized(&VALUES, 16),
            ),
            (AttrValue::string_array_sized(owned, 16).unwrap(), |b| {
                b.with_strings_sized(&VALUES, 16)
            }),
        ];
        for (attr, stage) in cases {
            let message = build_attr_message("a", &attr);
            let mut b = DatasetBuilder::new("d");
            stage(&mut b).unwrap();
            let (datatype, raw) = staged(&b);

            assert_eq!(datatype, message.datatype, "{attr:?}");
            assert_eq!(raw, message.raw_data, "{attr:?}");
        }
    }

    /// `with_shape` still decides the dataspace, as it does for every other data
    /// entry point; only the default comes from the value count.
    #[test]
    fn an_explicit_shape_survives_a_fixed_string_write() {
        let mut b = DatasetBuilder::new("d");
        b.with_shape(&[2, 2]);
        b.with_ascii_strings(&["a", "b", "c", "d"]).unwrap();

        assert_eq!(b.shape, Some(vec![2, 2]));
        assert_eq!(declared_width(&b, 4), 1);
    }

    // ---- Attribute widths a caller declares (issue #359) ----

    /// Each sized attribute constructor declares the width it was given, pads
    /// every element out to it, and keeps its own charset and dataspace kind.
    ///
    /// The width is read off the message and the stride off the bytes, because a
    /// datatype that says 64 over elements laid out at 2 is the defect this
    /// family exists to make impossible.
    #[test]
    fn a_declared_attribute_width_reaches_the_message_and_the_bytes() {
        let cases = [
            (
                AttrValue::ascii_string_sized("ok", 64).unwrap(),
                CharacterSet::Ascii,
                DataspaceType::Scalar,
                1,
            ),
            (
                AttrValue::string_sized("ok", 64).unwrap(),
                CharacterSet::Utf8,
                DataspaceType::Scalar,
                1,
            ),
            (
                AttrValue::ascii_string_array_sized(vec!["ok".into(), String::new()], 64).unwrap(),
                CharacterSet::Ascii,
                DataspaceType::Simple,
                2,
            ),
            (
                AttrValue::string_array_sized(vec!["ok".into(), String::new()], 64).unwrap(),
                CharacterSet::Utf8,
                DataspaceType::Simple,
                2,
            ),
        ];
        for (value, expected_charset, expected_space, count) in cases {
            let message = build_attr_message("a", &value);
            let Datatype::String {
                size,
                padding,
                charset,
            } = message.datatype.clone()
            else {
                panic!("expected a string datatype for {value:?}");
            };
            assert_eq!(size, 64, "{value:?}");
            assert_eq!(charset, expected_charset, "{value:?}");
            assert_eq!(padding, StringPadding::NullPad, "{value:?}");
            assert_eq!(message.dataspace.space_type, expected_space, "{value:?}");
            assert_eq!(message.raw_data.len(), 64 * count, "{value:?}");
            assert_eq!(&message.raw_data[..2], b"ok", "{value:?}");
            assert!(
                message.raw_data[2..].iter().all(|b| *b == 0),
                "everything past the value is padding, {value:?}"
            );
        }
    }

    /// The width the plain variants take is the content's own, which is what
    /// makes rewriting a slot shrink it — the behaviour issue #359 reported and
    /// the sized variants answer.
    #[test]
    fn a_plain_attribute_takes_the_content_width_and_a_sized_one_keeps_its_slot() {
        let shrunk = build_attr_message("a", &AttrValue::AsciiString("x".into()));
        assert!(matches!(shrunk.datatype, Datatype::String { size: 1, .. }));

        let kept = build_attr_message("a", &AttrValue::ascii_string_sized("x", 5).unwrap());
        assert!(matches!(kept.datatype, Datatype::String { size: 5, .. }));
    }

    /// A value past the declared width is refused when the value is built, not
    /// truncated when it is written — which is what lets every `set_attr` entry
    /// point below stay infallible. The index names the element that did not
    /// fit, so a long array does not have to be searched by hand.
    #[test]
    fn an_attribute_value_past_its_declared_width_is_refused() {
        assert!(matches!(
            AttrValue::ascii_string_sized("north", 2),
            Err(FormatError::FixedStringTooLong {
                index: 0,
                len: 5,
                width: 2
            })
        ));
        assert!(matches!(
            AttrValue::string_sized("mètre", 5),
            Err(FormatError::FixedStringTooLong {
                index: 0,
                len: 6,
                ..
            })
        ));
        assert!(matches!(
            AttrValue::ascii_string_array_sized(vec!["ab".into(), "north".into()], 4),
            Err(FormatError::FixedStringTooLong {
                index: 1,
                len: 5,
                width: 4
            })
        ));
        assert!(matches!(
            AttrValue::string_array_sized(vec!["ab".into(), "north".into()], 4),
            Err(FormatError::FixedStringTooLong { index: 1, .. })
        ));
    }

    /// A width of zero is refused for an attribute exactly as for a dataset. No
    /// value can make it legal: an empty one fits it, so the per-value check
    /// never fires.
    #[test]
    fn a_declared_attribute_width_of_zero_is_refused() {
        let refused = [
            AttrValue::ascii_string_sized("", 0).map(|_| ()),
            AttrValue::string_sized("", 0).map(|_| ()),
            AttrValue::ascii_string_array_sized(vec![], 0).map(|_| ()),
            AttrValue::string_array_sized(vec![], 0).map(|_| ()),
        ];
        for outcome in refused {
            assert!(
                matches!(outcome, Err(FormatError::ZeroFixedStringWidth)),
                "{outcome:?}"
            );
        }
    }
}
