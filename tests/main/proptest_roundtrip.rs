//! Property-based tests (issue #26).
//!
//! Two invariants that example-based tests can only spot-check but that
//! property testing exercises across a generated input space:
//!
//! 1. **Roundtrip identity.** For any in-range data and shape, writing a
//!    dataset and reading it back yields the original values, bit-exact (so
//!    `NaN`/`Inf` floats must roundtrip too), with the original shape and
//!    dtype. Covered for every numeric type the builder/reader support, in
//!    both contiguous and chunked+deflate layouts.
//!
//! 2. **Parser robustness.** Feeding arbitrary or corrupted bytes to the
//!    reader must return `Ok`/`Err` but never panic, index out of bounds, or
//!    overflow. proptest *shrinks* any offending input to its minimal form,
//!    which is exactly what makes parser bugs tractable to debug.
//!
//! When a property fails, proptest records the minimal reproducer under
//! `tests/proptest-regressions/` and replays it on every subsequent run. Those
//! files are deterministic seeds, not build artifacts: commit them so the
//! reproducer travels with the fix.

use hdf5_pure::{DType, File, FileBuilder, Group};
use proptest::prelude::*;

/// The 8-byte HDF5 magic signature. Mirrored here (rather than imported from
/// the crate-internal `signature` module) so the robustness tests can force the
/// reader past signature detection and into superblock/header parsing, where
/// the interesting failure modes live.
const HDF5_SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', b'\r', b'\n', 0x1A, b'\n'];

// ---------------------------------------------------------------------------
// Roundtrip identity
// ---------------------------------------------------------------------------

/// Strategy: a shape of 1..=3 dimensions, each extent 1..=6, paired with
/// exactly `product(shape)` elements drawn from `$elem`.
macro_rules! shaped {
    ($elem:expr) => {
        prop::collection::vec(1u64..=6, 1..=3).prop_flat_map(|shape| {
            let n = shape.iter().product::<u64>() as usize;
            (Just(shape), prop::collection::vec($elem, n..=n))
        })
    };
}

/// Generate a roundtrip property test for one numeric type. `$eq` compares the
/// readback against the input (plain `==` for integers, bit-exact for floats).
macro_rules! roundtrip_test {
    ($name:ident, $ty:ty, $with:ident, $read:ident, $dtype:expr, $eq:expr) => {
        proptest! {
            #[test]
            fn $name((shape, data) in shaped!(any::<$ty>())) {
                let mut b = FileBuilder::new();
                b.create_dataset("d").$with(&data).with_shape(&shape);
                let bytes = b.finish().expect("write");
                let file = File::from_bytes(bytes).expect("parse");
                let ds = file.dataset("d").expect("dataset");

                prop_assert_eq!(ds.shape().expect("shape"), shape);
                prop_assert_eq!(ds.dtype().expect("dtype"), $dtype);
                let got = ds.$read().expect("read");
                prop_assert!($eq(&got, &data), "roundtrip mismatch");
            }
        }
    };
}

fn int_eq<T: PartialEq>(a: &[T], b: &[T]) -> bool {
    a == b
}
fn f32_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn f64_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

roundtrip_test!(
    roundtrip_f64,
    f64,
    with_f64_data,
    read_f64,
    DType::F64,
    f64_eq
);
roundtrip_test!(
    roundtrip_f32,
    f32,
    with_f32_data,
    read_f32,
    DType::F32,
    f32_eq
);
roundtrip_test!(roundtrip_i8, i8, with_i8_data, read_i8, DType::I8, int_eq);
roundtrip_test!(
    roundtrip_i16,
    i16,
    with_i16_data,
    read_i16,
    DType::I16,
    int_eq
);
roundtrip_test!(
    roundtrip_i32,
    i32,
    with_i32_data,
    read_i32,
    DType::I32,
    int_eq
);
roundtrip_test!(
    roundtrip_i64,
    i64,
    with_i64_data,
    read_i64,
    DType::I64,
    int_eq
);
roundtrip_test!(roundtrip_u8, u8, with_u8_data, read_u8, DType::U8, int_eq);
roundtrip_test!(
    roundtrip_u16,
    u16,
    with_u16_data,
    read_u16,
    DType::U16,
    int_eq
);
roundtrip_test!(
    roundtrip_u32,
    u32,
    with_u32_data,
    read_u32,
    DType::U32,
    int_eq
);
roundtrip_test!(
    roundtrip_u64,
    u64,
    with_u64_data,
    read_u64,
    DType::U64,
    int_eq
);

proptest! {
    /// Chunked + deflate layout roundtrip (lossless integer path). Exercises
    /// the B-tree v1 chunk index and the filter pipeline, not just contiguous
    /// storage. Chunk dims are clamped to the shape so the request is valid.
    #[test]
    fn roundtrip_i32_chunked_deflate(
        (shape, chunk, data) in prop::collection::vec(1u64..=6, 1..=3).prop_flat_map(|shape| {
            let k = shape.len();
            let n = shape.iter().product::<u64>() as usize;
            (Just(shape), prop::collection::vec(1u64..=6, k..=k), prop::collection::vec(any::<i32>(), n..=n))
        }).prop_map(|(shape, seeds, data)| {
            let chunk = shape.iter().zip(&seeds).map(|(&d, &s)| s.min(d)).collect::<Vec<u64>>();
            (shape, chunk, data)
        })
    ) {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&data)
            .with_shape(&shape)
            .with_chunks(&chunk)
            .with_deflate(6);
        let bytes = b.finish().expect("write");
        let file = File::from_bytes(bytes).expect("parse");
        let ds = file.dataset("d").expect("dataset");

        prop_assert_eq!(ds.shape().expect("shape"), shape);
        prop_assert_eq!(ds.read_i32().expect("read"), data);
    }
}

// ---------------------------------------------------------------------------
// Parser robustness
// ---------------------------------------------------------------------------

/// Drive parsing one level deep without unwrapping: every call may legitimately
/// fail on a corrupt file, but none may panic. Deliberately shallow (no
/// recursion into nested groups) so a maliciously cyclic structure cannot hang
/// the test; the goal here is "parsing primitives never panic", not exhaustive
/// traversal.
fn exercise(file: &File) {
    let root: Group = file.root();
    if let Ok(names) = root.datasets() {
        for name in names.iter().take(16) {
            if let Ok(ds) = root.dataset(name) {
                let _ = ds.shape();
                let _ = ds.dtype();
                let _ = ds.read_f64();
                let _ = ds.read_i64();
                let _ = ds.read_u8();
                let _ = ds.read_string();
            }
        }
    }
    if let Ok(groups) = root.groups() {
        for name in groups.iter().take(16) {
            let _ = root.group(name);
        }
    }
}

/// A small but structurally complete valid file: root attribute, a contiguous
/// dataset, and a group containing a chunked+deflate dataset. Corrupting copies
/// of this drives far more of the parser than random bytes can.
fn valid_sample_file() -> Vec<u8> {
    let mut b = FileBuilder::new();
    b.set_attr("version", hdf5_pure::AttrValue::I64(2));
    b.create_dataset("contig")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
        .with_shape(&[2, 2]);
    let mut grp = b.create_group("g");
    grp.create_dataset("chunked")
        .with_i32_data(&[1, 2, 3, 4, 5, 6])
        .with_shape(&[6])
        .with_chunks(&[3])
        .with_deflate(4);
    let finished = grp.finish();
    b.add_group(finished);
    b.finish().expect("sample file must build")
}

proptest! {
    /// Arbitrary bytes: the common case returns `SignatureNotFound`, but the
    /// reader must never panic regardless.
    #[test]
    fn open_arbitrary_bytes_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..4096)) {
        if let Ok(file) = File::from_bytes(bytes) {
            exercise(&file);
        }
    }

    /// Arbitrary bytes behind a valid signature: forces the reader past
    /// signature detection into superblock and object-header parsing, where the
    /// interesting out-of-bounds/overflow paths are.
    #[test]
    fn open_signature_prefixed_bytes_never_panics(tail in prop::collection::vec(any::<u8>(), 0..4096)) {
        let mut bytes = HDF5_SIGNATURE.to_vec();
        bytes.extend_from_slice(&tail);
        if let Ok(file) = File::from_bytes(bytes) {
            exercise(&file);
        }
    }

    /// Corrupted valid file: flip random bytes and/or truncate a real file.
    /// This keeps the byte stream "shaped like" HDF5 so mutations land inside
    /// superblock fields, message headers, B-tree nodes, and chunk offsets,
    /// the regions where a careless parser over-reads.
    #[test]
    fn corrupted_valid_file_never_panics(
        muts in prop::collection::vec((any::<usize>(), any::<u8>()), 0..48),
        truncate in 0usize..=96,
    ) {
        let mut bytes = valid_sample_file();
        let len = bytes.len();
        for (idx, val) in muts {
            bytes[idx % len] = val;
        }
        bytes.truncate(len.saturating_sub(truncate));
        if let Ok(file) = File::from_bytes(bytes) {
            exercise(&file);
        }
    }
}
