//! Generic, type-parameterized dataset I/O via the `H5Element` bound:
//! `DatasetBuilder::with_data` and `Dataset::read`. See issue #53.
//!
//! These exercise the generic path on default features (no `ndarray`), so they
//! also prove the trait is available independently of that feature.

use hdf5_pure::{Dataset, Error, File, FileBuilder, H5Element};

/// Round-trip a flat slice through the *generic* write/read entry points and
/// assert the values come back unchanged. The function is itself generic over
/// the element type, which is the capability issue #53 asks for.
fn assert_round_trip<T>(name: &str, values: &[T])
where
    T: H5Element + PartialEq + std::fmt::Debug,
{
    let mut fb = FileBuilder::new();
    fb.create_dataset(name).with_data(values);
    let bytes = fb.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let back: Vec<T> = file.dataset(name).unwrap().read().unwrap();
    assert_eq!(values, back.as_slice(), "round trip mismatch for {name}");
}

#[test]
fn round_trips_every_supported_scalar() {
    assert_round_trip("f32", &[1.0f32, -2.5, 3.25, f32::MIN, f32::MAX]);
    assert_round_trip("f64", &[1.0f64, -2.5, 3.25, f64::MIN, f64::MAX]);
    assert_round_trip("i8", &[i8::MIN, -1, 0, 1, i8::MAX]);
    assert_round_trip("i16", &[i16::MIN, -1, 0, 1, i16::MAX]);
    assert_round_trip("i32", &[i32::MIN, -1, 0, 1, i32::MAX]);
    assert_round_trip("i64", &[i64::MIN, -1, 0, 1, i64::MAX]);
    assert_round_trip("u8", &[0u8, 1, 128, u8::MAX]);
    assert_round_trip("u16", &[0u16, 1, 128, u16::MAX]);
    assert_round_trip("u32", &[0u32, 1, 128, u32::MAX]);
    assert_round_trip("u64", &[0u64, 1, 128, u64::MAX]);
}

#[test]
fn generic_write_matches_type_specific_method() {
    // `with_data` must produce a byte-identical dataset to the hand-written
    // `with_i64_data`, since the generic path is meant to be a drop-in.
    let values: Vec<i64> = (-100..100).collect();

    let mut generic = FileBuilder::new();
    generic.create_dataset("d").with_data(&values);
    let a = generic.finish().unwrap();

    let mut specific = FileBuilder::new();
    specific.create_dataset("d").with_i64_data(&values);
    let b = specific.finish().unwrap();

    assert_eq!(a, b, "with_data must match with_i64_data byte-for-byte");
}

#[test]
fn read_is_type_inferred_and_turbofished() {
    let mut fb = FileBuilder::new();
    fb.create_dataset("v").with_data(&[10.0f64, 20.0, 30.0]);
    let file = File::from_bytes(fb.finish().unwrap()).unwrap();
    let ds = file.dataset("v").unwrap();

    // Inferred from the binding...
    let inferred: Vec<f64> = ds.read().unwrap();
    // ...or named with turbofish.
    let turbofished = ds.read::<f64>().unwrap();
    assert_eq!(inferred, vec![10.0, 20.0, 30.0]);
    assert_eq!(inferred, turbofished);
}

#[test]
fn with_data_preserves_an_explicit_multidimensional_shape() {
    // `with_data` only defaults the shape when none is set, so a chained
    // `with_shape` wins and the flat read returns row-major order.
    let values: Vec<i32> = (0..6).collect();
    let mut fb = FileBuilder::new();
    fb.create_dataset("m")
        .with_data(&values)
        .with_shape(&[2, 3]);
    let file = File::from_bytes(fb.finish().unwrap()).unwrap();
    let ds = file.dataset("m").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![2, 3]);
    assert_eq!(ds.read::<i32>().unwrap(), values);
}

#[test]
fn read_coerces_across_types_like_the_typed_methods() {
    // `read::<T>` requests delivery as `T`; it does not assert the stored type.
    // An i32 dataset read as f64 widens exactly as `read_f64` would.
    let mut fb = FileBuilder::new();
    fb.create_dataset("ints").with_data(&[1i32, 2, 3]);
    let file = File::from_bytes(fb.finish().unwrap()).unwrap();
    let ds = file.dataset("ints").unwrap();

    let as_f64: Vec<f64> = ds.read().unwrap();
    assert_eq!(as_f64, ds.read_f64().unwrap());
    assert_eq!(as_f64, vec![1.0, 2.0, 3.0]);
}

#[test]
fn generic_helpers_compose_into_user_code() {
    // The end-to-end scenario from the issue: one helper writes any scalar
    // type, another reads it back, both generic.
    fn store<T: H5Element>(fb: &mut FileBuilder, name: &str, values: &[T]) {
        fb.create_dataset(name).with_data(values);
    }
    fn load<T: H5Element>(file: &File, name: &str) -> Result<Vec<T>, Error> {
        file.dataset(name)?.read::<T>()
    }

    let mut fb = FileBuilder::new();
    store(&mut fb, "a", &[1u16, 2, 3]);
    let mut g = fb.create_group("grp");
    store_group(&mut g, "b", &[4.0f32, 5.0]);
    fb.add_group(g.finish());
    let file = File::from_bytes(fb.finish().unwrap()).unwrap();

    assert_eq!(load::<u16>(&file, "a").unwrap(), vec![1, 2, 3]);
    assert_eq!(load::<f32>(&file, "grp/b").unwrap(), vec![4.0, 5.0]);
}

/// Helper proving `with_data` is also reachable on datasets created inside a
/// group builder, not just at the file root.
fn store_group<T: H5Element>(g: &mut hdf5_pure::GroupBuilder, name: &str, values: &[T]) {
    g.create_dataset(name).with_data(values);
}

/// `_ = Dataset::read` keeps the import used even if the inherent method is
/// renamed; serves as a compile-time signature check.
#[allow(dead_code)]
fn signature_check(ds: &Dataset) -> Result<Vec<u8>, Error> {
    ds.read::<u8>()
}
