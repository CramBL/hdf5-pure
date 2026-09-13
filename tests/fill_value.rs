//! Configurable dataset fill values (issue #151).
//!
//! [`DatasetBuilder::with_fill_value`] records a user-defined fill value in the
//! dataset's Fill Value message, and [`Dataset::fill_value`] reads it back. These
//! tests pin the round trip across storage classes and datatypes, the "no fill
//! value" default, the datatype/size validation, and that a fill value set on an
//! `File::open_rw`-created dataset is honored.

use hdf5_pure::{Dataset, Error, File, FileBuilder, FormatError, H5Element, MaxExtent};
use tempfile::tempdir;

/// Build a file with `build`, serialize it, and return it parsed back, ready for
/// asserting on the round-tripped fill value.
fn round_trip_fill(build: impl FnOnce(&mut FileBuilder)) -> File {
    let mut fb = FileBuilder::new();
    build(&mut fb);
    let bytes = fb.finish().unwrap();
    File::from_bytes(bytes).unwrap()
}

#[test]
fn contiguous_fill_round_trips_and_keeps_data() {
    let file = round_trip_fill(|fb| {
        fb.create_dataset("d")
            .with_i32_data(&[10, 20, 30])
            .with_fill_value(-7_i32);
    });
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_value::<i32>().unwrap(), Some(-7));
    // The fill value must not disturb the stored data.
    assert_eq!(ds.read_i32().unwrap(), vec![10, 20, 30]);
}

#[test]
fn chunked_extensible_fill_round_trips() {
    let file = round_trip_fill(|fb| {
        fb.create_dataset("d")
            .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
            .with_shape(&[4])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[2])
            .with_fill_value(3.5_f64);
    });
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_value::<f64>().unwrap(), Some(3.5));
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn no_fill_value_reports_none() {
    let file = round_trip_fill(|fb| {
        fb.create_dataset("d").with_i32_data(&[1, 2, 3]);
    });
    // The crate's library-default fill message carries no user-defined value.
    assert_eq!(
        file.dataset("d").unwrap().fill_value::<i32>().unwrap(),
        None
    );
}

#[test]
fn fill_value_across_datatypes() {
    // Cover the byte widths and both the special u8/i8 paths and the LE-decoded
    // multi-byte paths.
    fn check<T: H5Element + PartialEq + std::fmt::Debug>(
        write: impl Fn(&mut hdf5_pure::DatasetBuilder),
        expect: T,
    ) {
        let mut fb = FileBuilder::new();
        write(fb.create_dataset("d"));
        let file = File::from_bytes(fb.finish().unwrap()).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().fill_value::<T>().unwrap(),
            Some(expect)
        );
    }
    check::<u8>(
        |d| {
            d.with_u8_data(&[1, 2]).with_fill_value(200_u8);
        },
        200,
    );
    check::<i8>(
        |d| {
            d.with_i8_data(&[1, 2]).with_fill_value(-5_i8);
        },
        -5,
    );
    check::<u16>(
        |d| {
            d.with_u16_data(&[1, 2]).with_fill_value(60000_u16);
        },
        60000,
    );
    check::<i16>(
        |d| {
            d.with_i16_data(&[1, 2]).with_fill_value(-30000_i16);
        },
        -30000,
    );
    check::<u32>(
        |d| {
            d.with_u32_data(&[1, 2]).with_fill_value(4_000_000_000_u32);
        },
        4_000_000_000,
    );
    check::<i32>(
        |d| {
            d.with_i32_data(&[1, 2]).with_fill_value(i32::MIN);
        },
        i32::MIN,
    );
    check::<u64>(
        |d| {
            d.with_u64_data(&[1, 2]).with_fill_value(u64::MAX);
        },
        u64::MAX,
    );
    check::<i64>(
        |d| {
            d.with_i64_data(&[1, 2]).with_fill_value(i64::MIN);
        },
        i64::MIN,
    );
    check::<f32>(
        |d| {
            d.with_f32_data(&[1.0, 2.0]).with_fill_value(-1.5_f32);
        },
        -1.5,
    );
    check::<f64>(
        |d| {
            d.with_f64_data(&[1.0, 2.0])
                .with_fill_value(f64::NEG_INFINITY);
        },
        f64::NEG_INFINITY,
    );
}

#[test]
fn fill_value_size_mismatch_is_refused() {
    // A fill value whose width differs from the dataset datatype is rejected at
    // build time, before any bytes are produced.
    let mut fb = FileBuilder::new();
    fb.create_dataset("d")
        .with_i32_data(&[1, 2, 3]) // 4-byte element
        .with_fill_value(9_u8); // 1-byte fill value
    let err = fb.finish().unwrap_err();
    assert!(
        matches!(
            err,
            Error::Format(FormatError::FillValueSizeMismatch {
                expected: 4,
                actual: 1
            })
        ),
        "expected a FillValueSizeMismatch, got {err:?}"
    );
}

#[test]
fn generic_fill_value_over_element_type() {
    // The generic path: a function generic over the element type sets both the
    // data and the fill value through `with_data` / `with_fill_value<T>`.
    fn store<T: H5Element>(fb: &mut FileBuilder, values: &[T], fill: T) {
        fb.create_dataset("d")
            .with_data(values)
            .with_fill_value(fill);
    }
    let mut fb = FileBuilder::new();
    store(&mut fb, &[1i64, 2, 3], 42i64);
    let file = File::from_bytes(fb.finish().unwrap()).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().fill_value::<i64>().unwrap(),
        Some(42)
    );
}

#[test]
fn fill_value_coerces_like_a_read() {
    // `fill_value::<T>` decodes with the same coercion rules as `read::<T>`:
    // an i32 fill value read as f64 widens.
    let file = round_trip_fill(|fb| {
        fb.create_dataset("d")
            .with_i32_data(&[1])
            .with_fill_value(-3_i32);
    });
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_value::<i32>().unwrap(), Some(-3));
    assert_eq!(ds.fill_value::<f64>().unwrap(), Some(-3.0));
}

#[test]
fn empty_dataset_carries_a_fill_value() {
    // A zero-element dataset owns no data block but still records its fill value.
    let file = round_trip_fill(|fb| {
        fb.create_dataset("d")
            .with_i32_data(&[])
            .with_shape(&[0])
            .with_fill_value(123_i32);
    });
    assert_eq!(
        file.dataset("d").unwrap().fill_value::<i32>().unwrap(),
        Some(123)
    );
}

#[test]
fn edit_session_created_dataset_carries_fill() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("edit.h5");

    // A base file to open for editing.
    {
        let mut fb = FileBuilder::new();
        fb.create_dataset("seed").with_i32_data(&[0]);
        fb.write(&p).unwrap();
    }

    // Add a dataset with a fill value through an edit session, then commit and
    // drop the session (releasing its exclusive lock) before reopening.
    {
        let s = File::open_rw(&p).unwrap();
        s.root()
            .create_dataset("added", |b| {
                b.with_f64_data(&[1.0, 2.0]).with_fill_value(-9.5_f64);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&p).unwrap();
    assert_eq!(
        file.dataset("added").unwrap().fill_value::<f64>().unwrap(),
        Some(-9.5)
    );
    assert_eq!(
        file.dataset("added").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0]
    );
}

#[test]
fn edit_session_fill_size_mismatch_is_refused() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("edit_bad.h5");
    {
        let mut fb = FileBuilder::new();
        fb.create_dataset("seed").with_i32_data(&[0]);
        fb.write(&p).unwrap();
    }
    let s = File::open_rw(&p).unwrap();
    // 2-byte fill on a 4-byte element
    let err = s
        .root()
        .create_dataset("added", |b| {
            b.with_i32_data(&[1, 2, 3]).with_fill_value(1_u16);
        })
        .unwrap_err();
    assert!(
        matches!(
            err,
            Error::Format(FormatError::FillValueSizeMismatch {
                expected: 4,
                actual: 2
            })
        ),
        "expected a FillValueSizeMismatch, got {err:?}"
    );
}

#[test]
fn write_dataset_refuses_a_fill_value() {
    // A value overwrite reuses the dataset's existing fill value, so a fill value
    // staged on the overwrite builder is refused rather than silently ignored —
    // consistent with write_dataset refusing attributes and layout changes.
    let dir = tempdir().unwrap();
    let p = dir.path().join("overwrite.h5");
    {
        let mut fb = FileBuilder::new();
        fb.create_dataset("d").with_i32_data(&[1, 2, 3]);
        fb.write(&p).unwrap();
    }
    let s = File::open_rw(&p).unwrap();
    let err = s
        .dataset("d")
        .unwrap()
        .write_staged(|b| {
            b.with_i32_data(&[4, 5, 6]).with_fill_value(-1_i32);
        })
        .unwrap_err();
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("fill value")),
        "expected a fill-value refusal, got {err:?}"
    );
    assert!(
        !s.has_staged_edits(),
        "a refused overwrite must not be left staged"
    );
}

/// A convenience shared with the crosscheck test: assert the fill value of a
/// dataset read from `file`.
fn assert_fill<T: H5Element + PartialEq + std::fmt::Debug>(ds: &Dataset, expect: Option<T>) {
    assert_eq!(ds.fill_value::<T>().unwrap(), expect);
}

#[test]
fn repack_preserves_fill_value() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("src.h5");
    let dst = dir.path().join("dst.h5");

    {
        let mut fb = FileBuilder::new();
        fb.create_dataset("contig")
            .with_i32_data(&[1, 2, 3])
            .with_fill_value(-7_i32);
        fb.create_dataset("chunked")
            .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
            .with_shape(&[4])
            .with_chunks(&[2])
            .with_fill_value(2.5_f64);
        fb.write(&src).unwrap();
    }

    hdf5_pure::repack(&src, &dst, &hdf5_pure::RepackOptions::default()).unwrap();

    let file = File::open(&dst).unwrap();
    assert_fill(&file.dataset("contig").unwrap(), Some(-7_i32));
    assert_fill(&file.dataset("chunked").unwrap(), Some(2.5_f64));
}
