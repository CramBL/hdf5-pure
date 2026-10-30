//! Configurable dataset fill values (issue #151).
//!
//! [`DatasetBuilder::with_fill_value`] records a user-defined fill value in the
//! dataset's Fill Value message, and [`Dataset::fill_value`] reads it back. These
//! tests pin the round trip across storage classes and datatypes, the "no fill
//! value" default, the datatype/size validation, and that a fill value set on an
//! `File::open_rw`-created dataset is honored.

use std::fmt::Debug;

use hdf5_pure::{
    Dataset, Datatype, DatatypeByteOrder, Error, File, FileBuilder, FillTime, FixedPointType,
    FormatError, H5Element, MaxExtent, RepackOptions,
};
use hdf5_pure_format::__private as format;
use rstest::rstest;

use test_util::data_layout;
use test_util::dataspace;
use test_util::image::Image;
use test_util::object_header::{Message, MessageType};
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;
use test_util_hdf5::dataset_writer::ScaleOffsetWriter;
use test_util_hdf5::file_builder;

#[test]
fn contiguous_fill_round_trips_and_keeps_data() {
    let file = file_builder::build_and_open(|fb| {
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
    let file = file_builder::build_and_open(|fb| {
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
    let file = file_builder::build_and_open(|fb| {
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
    fn check<T: H5Element + PartialEq + std::fmt::Debug>(
        write: impl Fn(&mut hdf5_pure::DatasetBuilder),
        expect: T,
    ) {
        let file = file_builder::build_and_open(|fb| {
            write(fb.create_dataset("d"));
        });
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
    let file = file_builder::build_and_open(|fb| {
        store(fb, &[1i64, 2, 3], 42i64);
    });
    assert_eq!(
        file.dataset("d").unwrap().fill_value::<i64>().unwrap(),
        Some(42)
    );
}

#[test]
fn fill_value_coerces_like_a_read() {
    // `fill_value::<T>` decodes with the same coercion rules as `read::<T>`:
    // an i32 fill value read as f64 widens.
    let file = file_builder::build_and_open(|fb| {
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
    let file = file_builder::build_and_open(|fb| {
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
    let dir = tempfile::tempdir().unwrap();
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
    let dir = tempfile::tempdir().unwrap();
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
    let dir = tempfile::tempdir().unwrap();
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
    let Error::EditUnsupported(actual) = err else {
        panic!("unexpected error: {err:?}");
    };
    assert_eq!(
        actual,
        "write_dataset overwrites values only; it cannot change the fill value (set it when the dataset is created)"
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
    let dir = tempfile::tempdir().unwrap();
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

#[test]
fn fill_time_defaults_to_if_set() {
    let file = file_builder::build_and_open(|fb| {
        fb.create_dataset("without").with_i32_data(&[1, 2, 3]);
        fb.create_dataset("with")
            .with_i32_data(&[1, 2, 3])
            .with_fill_value(42_i32);
    });
    assert_eq!(
        file.dataset("without").unwrap().fill_time().unwrap(),
        FillTime::IfSet
    );
    assert_eq!(
        file.dataset("with").unwrap().fill_time().unwrap(),
        FillTime::IfSet
    );
}

#[test]
fn fill_time_never_with_defined_fill_retains_metadata() {
    let mut fb = FileBuilder::new();
    fb.create_dataset("d")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
        .with_shape(&[4])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[2])
        .with_fill_value(3.5_f64);
    let mut bytes = fb.finish().unwrap();

    let header_at = bytes
        .windows(4)
        .rposition(|w| w == b"OHDR")
        .expect("dataset object header");
    let mut body = vec![3u8, 0x26, 8, 0, 0, 0];
    body.extend_from_slice(&3.5_f64.to_le_bytes());
    test_util::object_header::v2::replace_message(
        &mut bytes,
        header_at,
        test_util::object_header::MessageType::FILL_VALUE,
        &body,
    );

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_time().unwrap(), FillTime::Never);
    assert_eq!(ds.fill_value::<f64>().unwrap(), Some(3.5));
}

#[test]
fn fill_time_invalid_flag_is_rejected() {
    let mut fb = FileBuilder::new();
    fb.create_dataset("d")
        .with_f64_data(&[1.0, 2.0])
        .with_fill_value(3.5_f64);
    let mut bytes = fb.finish().unwrap();

    let header_at = bytes
        .windows(4)
        .rposition(|w| w == b"OHDR")
        .expect("dataset object header");
    let mut body = vec![3u8, 0x2e, 8, 0, 0, 0];
    body.extend_from_slice(&3.5_f64.to_le_bytes());
    test_util::object_header::v2::replace_message(
        &mut bytes,
        header_at,
        test_util::object_header::MessageType::FILL_VALUE,
        &body,
    );

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("d").unwrap();
    let err = ds.fill_time().unwrap_err();
    let Error::Format(FormatError::InvalidFillTime(code)) = err else {
        panic!("expected InvalidFillTime, got {err:?}");
    };
    assert_eq!(code, 3);
    let err_val = ds.fill_value::<f64>().unwrap_err();
    let Error::Format(FormatError::InvalidFillTime(code)) = err_val else {
        panic!("expected InvalidFillTime, got {err_val:?}");
    };
    assert_eq!(code, 3);
}

#[rstest]
#[case::on_allocation(FillTime::Alloc)]
#[case::never(FillTime::Never)]
#[case::if_set(FillTime::IfSet)]
fn fill_time_selection_round_trips(#[case] fill_time: FillTime) {
    let file = file_builder::build_and_open(|fb| {
        fb.create_dataset("contig")
            .with_i32_data(&[1, 2, 3])
            .with_fill_time(fill_time);
        fb.create_dataset("chunked")
            .with_i32_data(&[4, 5, 6, 7])
            .with_shape(&[4])
            .with_chunks(&[2])
            .with_fill_time(fill_time);
    });
    assert_eq!(
        file.dataset("contig").unwrap().fill_time().unwrap(),
        fill_time
    );
    assert_eq!(
        file.dataset("chunked").unwrap().fill_time().unwrap(),
        fill_time
    );
    assert_eq!(
        file.dataset("contig").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(
        file.dataset("chunked").unwrap().read_i32().unwrap(),
        vec![4, 5, 6, 7]
    );
}

#[test]
fn fill_time_never_with_defined_fill_round_trips_through_builder() {
    let file = file_builder::build_and_open(|fb| {
        fb.create_dataset("d")
            .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
            .with_shape(&[4])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[2])
            .with_fill_value(3.5_f64)
            .with_fill_time(FillTime::Never);
    });
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_time().unwrap(), FillTime::Never);
    assert_eq!(ds.fill_value::<f64>().unwrap(), Some(3.5));
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn chunk_edge_padding_is_zeroed_under_fill_time_never() {
    let mut b = FileBuilder::new();
    b.create_dataset("default_fill")
        .with_i32_data(&[1, 2, 3, 4, 5])
        .with_shape(&[5])
        .with_chunks(&[4])
        .with_fill_value(99_i32);
    b.create_dataset("never_fill")
        .with_i32_data(&[1, 2, 3, 4, 5])
        .with_shape(&[5])
        .with_chunks(&[4])
        .with_fill_value(99_i32)
        .with_fill_time(FillTime::Never);
    let bytes = b.finish().unwrap();

    let file = File::from_bytes(bytes.clone()).unwrap();
    let default_ds = file.dataset("default_fill").unwrap();
    let never_ds = file.dataset("never_fill").unwrap();

    assert_eq!(default_ds.fill_time().unwrap(), FillTime::IfSet);
    assert_eq!(default_ds.fill_value::<i32>().unwrap(), Some(99));
    let default_chunks = default_ds.chunks().unwrap();
    let default_c1 = &default_chunks[1];
    let default_raw = &bytes
        [default_c1.address as usize..(default_c1.address + default_c1.storage_size) as usize];
    assert_eq!(
        default_raw,
        &[5, 0, 0, 0, 99, 0, 0, 0, 99, 0, 0, 0, 99, 0, 0, 0]
    );

    assert_eq!(never_ds.fill_time().unwrap(), FillTime::Never);
    assert_eq!(never_ds.fill_value::<i32>().unwrap(), Some(99));
    let never_chunks = never_ds.chunks().unwrap();
    let never_c1 = &never_chunks[1];
    let never_raw =
        &bytes[never_c1.address as usize..(never_c1.address + never_c1.storage_size) as usize];
    assert_eq!(never_raw, &[5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn edit_session_created_dataset_preserves_fill_time() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("edit_fill_time.h5");

    {
        let mut fb = FileBuilder::new();
        fb.create_dataset("seed").with_i32_data(&[0]);
        fb.write(&p).unwrap();
    }

    {
        let s = File::open_rw(&p).unwrap();
        s.root()
            .create_dataset("never", |b| {
                b.with_f64_data(&[1.0, 2.0])
                    .with_fill_value(-9.5_f64)
                    .with_fill_time(FillTime::Never);
            })
            .unwrap();
        s.root()
            .create_dataset("alloc", |b| {
                b.with_i32_data(&[10, 20]).with_fill_time(FillTime::Alloc);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&p).unwrap();
    let never_ds = file.dataset("never").unwrap();
    assert_eq!(never_ds.fill_time().unwrap(), FillTime::Never);
    assert_eq!(never_ds.fill_value::<f64>().unwrap(), Some(-9.5));
    assert_eq!(never_ds.read_f64().unwrap(), vec![1.0, 2.0]);

    let alloc_ds = file.dataset("alloc").unwrap();
    assert_eq!(alloc_ds.fill_time().unwrap(), FillTime::Alloc);
    assert_eq!(alloc_ds.read_i32().unwrap(), vec![10, 20]);
}

#[rstest]
#[case::default_policy(FillTime::IfSet)]
#[case::allocate(FillTime::Alloc)]
#[case::never(FillTime::Never)]
fn write_dataset_rejects_an_explicit_fill_time(#[case] fill_time: FillTime) {
    let p = test_util::temp::temp_path("overwrite_fill_time.h5");
    {
        let mut fb = FileBuilder::new();
        fb.create_dataset("d").with_i32_data(&[1, 2, 3]);
        fb.write(&p).unwrap();
    }
    let before = std::fs::read(&p).unwrap();
    let s = File::open_rw(&p).unwrap();
    let err = s
        .dataset("d")
        .unwrap()
        .write_staged(|b| {
            b.with_i32_data(&[4, 5, 6]).with_fill_time(fill_time);
        })
        .unwrap_err();
    let Error::EditUnsupported(reason) = err else {
        panic!("expected EditUnsupported, got {err:?}");
    };
    assert_eq!(
        reason,
        "write_dataset overwrites values only; it cannot change the fill time (set it when the dataset is created)"
    );
    assert!(
        !s.has_staged_edits(),
        "a rejected overwrite must not be left staged"
    );
    s.commit().unwrap();
    drop(s);
    assert_eq!(std::fs::read(&p).unwrap(), before);
}

#[test]
fn repack_preserves_fill_time_policies() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src_policies.h5");
    let dst = dir.path().join("dst_policies.h5");

    {
        let mut fb = FileBuilder::new();
        fb.create_dataset("never")
            .with_i32_data(&[1, 2, 3])
            .with_fill_time(FillTime::Never);
        fb.create_dataset("never_with_value")
            .with_i32_data(&[4, 5, 6])
            .with_fill_value(-7_i32)
            .with_fill_time(FillTime::Never);
        fb.create_dataset("alloc")
            .with_f64_data(&[1.0, 2.0])
            .with_fill_time(FillTime::Alloc);
        fb.create_dataset("if_set")
            .with_f64_data(&[3.0, 4.0])
            .with_fill_value(9.5_f64)
            .with_fill_time(FillTime::IfSet);
        fb.create_dataset("default_policy")
            .with_i32_data(&[7, 8, 9]);
        fb.write(&src).unwrap();
    }

    hdf5_pure::repack(&src, &dst, &RepackOptions::default()).unwrap();

    let file = File::open(&dst).unwrap();
    let never_ds = file.dataset("never").unwrap();
    assert_eq!(never_ds.fill_time().unwrap(), FillTime::Never);
    assert_eq!(never_ds.fill_value::<i32>().unwrap(), None);

    let never_val_ds = file.dataset("never_with_value").unwrap();
    assert_eq!(never_val_ds.fill_time().unwrap(), FillTime::Never);
    assert_eq!(never_val_ds.fill_value::<i32>().unwrap(), Some(-7));

    let alloc_ds = file.dataset("alloc").unwrap();
    assert_eq!(alloc_ds.fill_time().unwrap(), FillTime::Alloc);

    let if_set_ds = file.dataset("if_set").unwrap();
    assert_eq!(if_set_ds.fill_time().unwrap(), FillTime::IfSet);
    assert_eq!(if_set_ds.fill_value::<f64>().unwrap(), Some(9.5));

    let default_ds = file.dataset("default_policy").unwrap();
    assert_eq!(default_ds.fill_time().unwrap(), FillTime::IfSet);
    assert_eq!(default_ds.fill_value::<i32>().unwrap(), None);
}

#[rstest]
#[case::partial(&[0])]
#[case::extra_partial(&[0, 42, 0])]
#[case::two_elements(&[0, 42, 0, 43])]
fn rejects_scalar_fill_size(#[case] raw: &[u8]) {
    let datatype = Datatype::FixedPoint(
        FixedPointType::new(2, DatatypeByteOrder::LittleEndian, false, 0, 16).unwrap(),
    );
    let fill = format::serialize_fill_value_message_v3(FillTime::IfSet, Some(raw)).unwrap();
    let image = Image::starting_with(&vec![0; Superblock::new(Widths::EIGHT).build().len()]);
    let bytes = image.build_with_dataset(
        "d",
        [
            Message::new(MessageType::DATASPACE, &dataspace::scalar()),
            Message::new(
                MessageType::DATATYPE,
                &format::serialize_datatype(&datatype),
            ),
            Message::new(MessageType::FILL_VALUE, &fill),
            Message::new(
                MessageType::DATA_LAYOUT,
                &data_layout::compact(&42u16.to_le_bytes()),
            ),
        ],
    );
    let file = File::from_bytes(bytes).unwrap();
    let error = file.dataset("d").unwrap().fill_value::<u8>().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected numeric format error, got {error:?}");
    };
    assert_eq!(
        error,
        FormatError::DataSizeMismatch {
            expected: 2,
            actual: raw.len()
        }
    );
}

#[rstest]
#[case::negative_zero(0x8000_0000_0000_0000)]
#[case::subnormal(1)]
#[case::infinity(0x7ff0_0000_0000_0000)]
#[case::nan_payload(0x7ff8_0000_1234_5678)]
fn preserves_scalar_fill_bits(#[case] bits: u64) {
    let value = f64::from_bits(bits);
    let file = file_builder::build_and_open(|builder| {
        builder
            .create_dataset("d")
            .with_f64_data(&[value])
            .with_fill_value(value);
    });
    let dataset = file.dataset("d").unwrap();
    assert_eq!(
        dataset.fill_value::<f64>().unwrap().unwrap().to_bits(),
        bits
    );
    assert_eq!(dataset.read_f64().unwrap()[0].to_bits(), bits);
}

#[rstest]
#[case::negative_zero(0x8000_0000)]
#[case::subnormal(1)]
#[case::infinity(0x7f80_0000)]
#[case::nan_payload(0x7fc0_1234)]
fn preserves_scalar_f32_fill_bits(#[case] bits: u32) {
    let value = f32::from_bits(bits);
    let file = file_builder::build_and_open(|builder| {
        builder
            .create_dataset("d")
            .with_f32_data(&[value])
            .with_fill_value(value);
    });
    let dataset = file.dataset("d").unwrap();
    assert_eq!(
        dataset.fill_value::<f32>().unwrap().unwrap().to_bits(),
        bits
    );
    assert_eq!(dataset.read_f32().unwrap()[0].to_bits(), bits);
}

#[rstest]
#[case::f32(99.125_f32)]
#[case::f64(99.125_f64)]
fn scale_offset_preserves_explicit_fill_values_under_every_fill_policy<T>(
    #[case] fill: T,
    #[values(FillTime::IfSet, FillTime::Alloc, FillTime::Never)] fill_time: FillTime,
    #[values(
        ScaleOffsetWriter::Buffered,
        ScaleOffsetWriter::Streaming,
        ScaleOffsetWriter::CreateAppend,
        ScaleOffsetWriter::CreateAppendStaged,
        ScaleOffsetWriter::StagedAppend
    )]
    writer: ScaleOffsetWriter,
) where
    T: H5Element + From<f32> + PartialEq + Debug,
{
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("scale_offset.h5");
    let values = [T::from(1.0), fill, T::from(12.0), T::from(2.0)];
    writer.write(&path, &values, fill, fill_time);
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("col").unwrap();
    assert_eq!(dataset.shape().unwrap(), [4]);
    assert_eq!(dataset.chunk_shape().unwrap(), Some(vec![4]));
    assert_eq!(dataset.chunks().unwrap().len(), 1);
    assert_eq!(dataset.fill_value::<T>().unwrap(), Some(fill));
    assert_eq!(dataset.fill_time().unwrap(), fill_time);
    assert_eq!(dataset.read::<T>().unwrap(), values);
}
