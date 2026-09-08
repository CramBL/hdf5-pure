//! Guards the `#[non_exhaustive]` seals on the public API against silent removal.
//!
//! Adding a variant or field to a public type is a breaking change unless the
//! type is sealed, so a seal deleted by accident is a break that surfaces only
//! at release time. Nothing else in the suite notices: an exhaustive match and a
//! struct literal both keep compiling when a seal *appears*, so ordinary tests
//! are blind in the direction that matters.
//!
//! Two mechanisms, because Rust only offers a positive test for one of them.
//!
//! **Enums** are checked semantically, by `#![deny(unreachable_patterns)]` plus a
//! match that names every variant *and* a `_` arm. That arm is reachable only
//! while the enum is sealed; delete the seal and the `_` becomes unreachable and
//! this file fails to compile. Adding a variant later needs no change here — the
//! `_` simply keeps covering it — so the guard costs nothing to carry.
//!
//! **Structs and sealed enum *variants*** have no equivalent positive signal (a
//! sealed struct cannot be constructed or destructured from here at all, and a
//! redundant `..` is not linted), so their seals are asserted against the source
//! text instead. Crude, but it fails loudly on deletion, which is the whole job.
//!
//! `compile_fail` doctests were the obvious alternative and are a trap: rustdoc
//! ignores the `E0004`-style error-code pin, so such a test passes when the
//! snippet fails for *any* reason — including a rename that stops it exercising
//! the seal at all. It would rot into a vacuous pass with no signal.
//!
//! `Error`, `FormatError`, and the introspection types from #149/#150 are sealed
//! but not listed here: their variant sets are large and were sealed before this
//! convention was written down. Extend the enum guard if they ever need it.

#![deny(unreachable_patterns)]

use hdf5_pure::{
    AttrValue, CompoundMember, DType, Datatype, EditBacking, FileSpaceInfo, LibVer, MemoryStrategy,
    Object, ReferenceType, SyncPolicy, mat::MatClass,
};

/// Every sealed enum, matched over its full variant set plus a `_` arm. This
/// compiles only while each one is `#[non_exhaustive]`.
#[test]
fn sealed_enums_still_require_a_wildcard_arm() {
    fn libver(v: LibVer) {
        match v {
            LibVer::Earliest | LibVer::V18 | LibVer::V110 | LibVer::V112 | LibVer::V114 => {}
            _ => {}
        }
    }

    fn datatype(d: &Datatype) {
        match d {
            Datatype::FixedPoint { .. }
            | Datatype::FloatingPoint { .. }
            | Datatype::Time { .. }
            | Datatype::String { .. }
            | Datatype::BitField { .. }
            | Datatype::Opaque { .. }
            | Datatype::Compound { .. }
            | Datatype::Reference { .. }
            | Datatype::Enumeration { .. }
            | Datatype::VariableLength { .. }
            | Datatype::Array { .. } => {}
            _ => {}
        }
    }

    fn memory_strategy(m: MemoryStrategy) {
        match m {
            MemoryStrategy::Bounded | MemoryStrategy::Auto | MemoryStrategy::Mirrored => {}
            _ => {}
        }
    }

    fn sync_policy(p: SyncPolicy) {
        match p {
            SyncPolicy::Always | SyncPolicy::OnClose => {}
            _ => {}
        }
    }

    fn edit_backing(b: EditBacking) {
        match b {
            EditBacking::Bounded | EditBacking::Mirrored => {}
            _ => {}
        }
    }

    fn reference_type(r: &ReferenceType) {
        match r {
            ReferenceType::Object | ReferenceType::DatasetRegion => {}
            _ => {}
        }
    }

    fn attr_value(a: &AttrValue) {
        match a {
            AttrValue::F64(_)
            | AttrValue::F64Array(_)
            | AttrValue::I32(_)
            | AttrValue::I64(_)
            | AttrValue::I64Array(_)
            | AttrValue::U32(_)
            | AttrValue::U64(_)
            | AttrValue::String(_)
            | AttrValue::StringArray(_)
            | AttrValue::AsciiString(_)
            | AttrValue::AsciiStringArray(_)
            | AttrValue::VarLenAsciiCharArray(_) => {}
            _ => {}
        }
    }

    fn dtype(d: &DType) {
        match d {
            DType::F32
            | DType::F64
            | DType::I8
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::U8
            | DType::U16
            | DType::U32
            | DType::U64
            | DType::String
            | DType::Compound(_)
            | DType::Enum(_)
            | DType::Array(_, _)
            | DType::VariableLengthString
            | DType::ObjectReference
            | DType::Other(_) => {}
            _ => {}
        }
    }

    fn object(o: &Object) {
        match o {
            Object::Group(_) | Object::Dataset(_) => {}
            _ => {}
        }
    }

    fn mat_class(c: MatClass) {
        match c {
            MatClass::Double
            | MatClass::Single
            | MatClass::Int8
            | MatClass::Int16
            | MatClass::Int32
            | MatClass::Int64
            | MatClass::UInt8
            | MatClass::UInt16
            | MatClass::UInt32
            | MatClass::UInt64
            | MatClass::Char
            | MatClass::Logical
            | MatClass::Struct
            | MatClass::Cell => {}
            _ => {}
        }
    }

    // Referenced so a rename cannot leave the guards above silently unused.
    libver(LibVer::LATEST);
    memory_strategy(MemoryStrategy::Bounded);
    sync_policy(SyncPolicy::Always);
    edit_backing(EditBacking::Bounded);
    reference_type(&ReferenceType::Object);
    attr_value(&AttrValue::I32(1));
    dtype(&DType::F64);
    mat_class(MatClass::Double);
    datatype(&Datatype::Time {
        size: 8,
        byte_order: hdf5_pure::DatatypeByteOrder::LittleEndian,
        bit_precision: 64,
    });
    let _: fn(&Object) = object;
}

#[cfg(feature = "provenance")]
#[test]
fn sealed_verify_result_still_requires_a_wildcard_arm() {
    use hdf5_pure::VerifyResult;
    fn verify(v: &VerifyResult) {
        match v {
            VerifyResult::Ok | VerifyResult::Mismatch { .. } | VerifyResult::NoHash => {}
            _ => {}
        }
    }
    verify(&VerifyResult::Ok);
}

/// Whether the declaration `decl` in `file` carries `#[non_exhaustive]` among
/// the attributes and doc comments directly above it.
///
/// Panics if `decl` is not in the file at all: a guard that quietly finds
/// nothing is a guard that stopped checking.
fn sealed_in_source(file: &str, decl: &str) -> bool {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(root.join(file))
        .unwrap_or_else(|e| panic!("cannot read {file}: {e}"));
    let at = src
        .find(decl)
        .unwrap_or_else(|| panic!("{file} no longer declares `{decl}` — update this guard"));
    src[..at]
        .lines()
        .rev()
        .take_while(|l| {
            let t = l.trim();
            t.starts_with('#') || t.starts_with("///") || t.starts_with("//")
        })
        .any(|l| l.trim() == "#[non_exhaustive]")
}

/// The sealed structs, asserted against the source text. A sealed struct cannot
/// be constructed or destructured from an external crate at all, so there is no
/// expression that compiles only while the seal is present — unlike the enums
/// above, this has to read the declaration.
#[test]
fn sealed_structs_keep_their_attribute() {
    // Named so a rename breaks the build here rather than silently skipping the
    // assertion below.
    fn _live(info: &FileSpaceInfo, member: &CompoundMember) -> usize {
        info.manager_addrs.len() + member.name.len()
    }

    let expected = [
        ("src/file_space_info.rs", "pub struct FileSpaceInfo {"),
        ("src/datatype.rs", "pub struct CompoundMember {"),
        ("src/datatype.rs", "pub struct EnumMember {"),
        ("src/mat/opaque.rs", "pub struct MatDatetime {"),
        ("src/mat/opaque.rs", "pub struct MatDuration {"),
        ("src/mat/opaque.rs", "pub struct MatCategorical {"),
        ("src/mat/opaque.rs", "pub struct MatEnum {"),
    ];

    for (file, decl) in expected {
        assert!(
            sealed_in_source(file, decl),
            "`{decl}` in {file} lost its #[non_exhaustive]: adding a field to it is now a \
             breaking change for every caller. Restore the attribute, or drop this entry and \
             record the break in CHANGELOG.md."
        );
    }
}

/// The sealed *variants*, which are a stronger claim than a sealed struct: their
/// seal is what stops an [`AttrValue`] being written by hand past the checked
/// constructor that is supposed to be the only way to build one (issue #359).
/// Lose it and a caller can declare a width too small for the value beside it,
/// which the infallible write path has no way to refuse.
///
/// Asserted against the source text for the same reason as the structs: a
/// pattern needs `..` while the seal is present, and keeps compiling with a
/// redundant `..` once it is gone, so nothing here can test it positively.
#[test]
fn sealed_variants_keep_their_attribute() {
    // Named so a rename breaks the build here rather than silently skipping the
    // assertion below.
    fn _live(v: &AttrValue) -> Option<u32> {
        match v {
            AttrValue::AsciiStringSized { width, .. }
            | AttrValue::AsciiStringArraySized { width, .. }
            | AttrValue::StringSized { width, .. }
            | AttrValue::StringArraySized { width, .. } => Some(width.get()),
            _ => None,
        }
    }

    for decl in [
        "    StringSized {",
        "    StringArraySized {",
        "    AsciiStringSized {",
        "    AsciiStringArraySized {",
    ] {
        assert!(
            sealed_in_source("src/type_builders.rs", decl),
            "`{decl}` in src/type_builders.rs lost its #[non_exhaustive]: a caller can now build \
             one past `AttrValue::ascii_string_sized` and its siblings, which is where a value \
             too long for its declared width is refused. Restore the attribute."
        );
    }
}
