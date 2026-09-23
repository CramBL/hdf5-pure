//! Reading MATLAB MCOS opaque value classes (`datetime`, `duration`,
//! `categorical`) and the lossless `Opaque` fallback for classes without a
//! dedicated decoder.
//!
//! There is no MATLAB available in CI, so these fixtures are assembled in Rust
//! to the documented `#subsystem#/FileWrapper__` byte layout (cross-validated
//! against the `matio`, `MatFileHandler`, and `foreverallama` parsers and
//! against this crate's own writer). They exercise the full read path: opaque
//! parent metadata -> object/class/property tables -> heap-cell resolution ->
//! per-class decode. The structural parser is additionally checked against the
//! production string writer's blob in `mcos_reader`'s unit tests, which is
//! real-MATLAB-derived rather than self-generated.

#![cfg(feature = "serde")]

use hdf5_pure::mat::{self, MatCategorical, MatDatetime, MatDuration};
use serde::Deserialize;
use test_util::mcos::{self, FileWrapper};
use test_util_hdf5::mat_file::{self, Cell};

#[test]
fn datetime_decodes_millis_and_sub_ms() {
    // datetime with one object whose `data` property is a 1x2 complex double:
    // 1000 ms (whole) and 2000 ms + 0.5 ms sub-millisecond.
    let blob = FileWrapper::new(&["datetime", "data"])
        .class(0, 1) // class 1 = "datetime"
        .object(1, 0, 1) // object 1: class 1, no saveobj block, normal-segment block 1
        .normal_block(&[(2, mcos::HEAP, 0)]) // block 1: data -> heap cell 0 (= MCOS cell 2)
        .build();
    let meta = mcos::object_metadata(1, 1);
    let bytes = mat_file::object_file(
        "ts",
        "datetime",
        &meta,
        &[
            ("blob", Cell::Blob(blob)),
            ("canon", Cell::Empty),
            ("data", Cell::ComplexF64(vec![(1000.0, 0.0), (2000.0, 0.5)])),
        ],
        &[],
    );

    #[derive(Deserialize)]
    struct Doc {
        ts: MatDatetime,
    }
    let doc: Doc = mat::from_bytes(&bytes).expect("decode datetime");
    assert_eq!(doc.ts.millis_utc, vec![1000.0, 2000.0]);
    assert_eq!(doc.ts.sub_ms, vec![0.0, 0.5]);
    assert_eq!(doc.ts.nanoseconds(), vec![1_000_000_000.0, 2_000_500_000.0]);
}

#[test]
fn datetime_carries_format_string() {
    let blob = FileWrapper::new(&["datetime", "data", "fmt"])
        .class(0, 1)
        .object(1, 0, 1)
        .normal_block(&[(2, mcos::HEAP, 0), (3, mcos::HEAP, 1)]) // `data` -> cell 2, `fmt` -> cell 3
        .build();
    let meta = mcos::object_metadata(1, 1);
    let bytes = mat_file::object_file(
        "t",
        "datetime",
        &meta,
        &[
            ("blob", Cell::Blob(blob)),
            ("canon", Cell::Empty),
            ("data", Cell::ComplexF64(vec![(0.0, 0.0)])),
            ("fmt", Cell::Char("uuuu-MM-dd".to_owned())),
        ],
        &[],
    );

    #[derive(Deserialize)]
    struct Doc {
        t: MatDatetime,
    }
    let doc: Doc = mat::from_bytes(&bytes).expect("decode datetime with fmt");
    assert_eq!(doc.t.millis_utc, vec![0.0]);
    assert_eq!(doc.t.fmt.as_deref(), Some("uuuu-MM-dd"));
}

#[test]
fn duration_decodes_milliseconds() {
    let blob = FileWrapper::new(&["duration", "millis"])
        .class(0, 1)
        .object(1, 0, 1)
        .normal_block(&[(2, mcos::HEAP, 0)])
        .build();
    let meta = mcos::object_metadata(1, 1);
    let bytes = mat_file::object_file(
        "elapsed",
        "duration",
        &meta,
        &[
            ("blob", Cell::Blob(blob)),
            ("canon", Cell::Empty),
            ("millis", Cell::F64(vec![1000.0, 2000.0, 60_000.0])),
        ],
        &[],
    );

    #[derive(Deserialize)]
    struct Doc {
        elapsed: MatDuration,
    }
    let doc: Doc = mat::from_bytes(&bytes).expect("decode duration");
    assert_eq!(doc.elapsed.millis, vec![1000.0, 2000.0, 60_000.0]);
    assert_eq!(doc.elapsed.seconds(), vec![1.0, 2.0, 60.0]);
}

#[test]
fn categorical_decodes_codes_categories_and_flags() {
    // codes 1-based with a 0 = <undefined>; two categories; ordinal, unprotected.
    let blob = FileWrapper::new(&[
        "categorical",
        "codes",
        "categoryNames",
        "isOrdinal",
        "isProtected",
    ])
    .class(0, 1)
    .object(1, 0, 1)
    .normal_block(&[
        (2, mcos::HEAP, 0),   // codes -> cell 2
        (3, mcos::HEAP, 1),   // `categoryNames` -> cell 3
        (4, mcos::INLINE, 1), // `isOrdinal` = true
        (5, mcos::INLINE, 0), // `isProtected` = false
    ])
    .build();
    let meta = mcos::object_metadata(1, 1);
    let bytes = mat_file::object_file(
        "grade",
        "categorical",
        &meta,
        &[
            ("blob", Cell::Blob(blob)),
            ("canon", Cell::Empty),
            ("codes", Cell::U8(vec![1, 2, 1, 0])),
            (
                "catnames",
                Cell::Refs(vec!["#refs#/cat0".into(), "#refs#/cat1".into()]),
            ),
        ],
        &[
            ("cat0", Cell::Char("Low".to_owned())),
            ("cat1", Cell::Char("High".to_owned())),
        ],
    );

    #[derive(Deserialize)]
    struct Doc {
        grade: MatCategorical,
    }
    let doc: Doc = mat::from_bytes(&bytes).expect("decode categorical");
    assert_eq!(doc.grade.codes, vec![1, 2, 1, 0]);
    assert_eq!(
        doc.grade.categories,
        vec!["Low".to_owned(), "High".to_owned()]
    );
    assert!(doc.grade.is_ordinal);
    assert!(!doc.grade.is_protected);
    assert_eq!(
        doc.grade.labels(),
        vec![
            Some("Low".to_owned()),
            Some("High".to_owned()),
            Some("Low".to_owned()),
            None,
        ]
    );
}

#[test]
fn unknown_opaque_class_is_lossless() {
    // A class without a dedicated decoder (here a namespaced `containers.Map`)
    // surfaces its raw properties so it still deserializes as a struct.
    let blob = FileWrapper::new(&["containers", "Map", "value"])
        .class(1, 2) // class 1 = namespace "containers" + name "Map"
        .object(1, 0, 1)
        .normal_block(&[(3, mcos::HEAP, 0)]) // value -> cell 2
        .build();
    let meta = mcos::object_metadata(1, 1);
    let bytes = mat_file::object_file(
        "m",
        "containers.Map",
        &meta,
        &[
            ("blob", Cell::Blob(blob)),
            ("canon", Cell::Empty),
            ("value", Cell::F64(vec![42.0])),
        ],
        &[],
    );

    #[derive(Deserialize)]
    struct Inner {
        value: f64,
    }
    #[derive(Deserialize)]
    struct Doc {
        m: Inner,
    }
    let doc: Doc = mat::from_bytes(&bytes).expect("decode unknown opaque as struct");
    assert_eq!(doc.m.value, 42.0);
}

#[test]
fn dangling_heap_reference_is_an_error_not_a_panic() {
    // A property points at heap value 5 (MCOS cell 7), which does not exist.
    let blob = FileWrapper::new(&["datetime", "data"])
        .class(0, 1)
        .object(1, 0, 1)
        .normal_block(&[(2, mcos::HEAP, 5)])
        .build();
    let meta = mcos::object_metadata(1, 1);
    let bytes = mat_file::object_file(
        "ts",
        "datetime",
        &meta,
        &[("blob", Cell::Blob(blob)), ("canon", Cell::Empty)],
        &[],
    );

    #[derive(Deserialize)]
    struct Doc {
        #[allow(dead_code)]
        ts: MatDatetime,
    }
    let result: Result<Doc, _> = mat::from_bytes(&bytes);
    assert!(result.is_err(), "dangling heap reference must error");
}
