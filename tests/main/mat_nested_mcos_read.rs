//! Nested / embedded MCOS object reference decoding, validated against a real
//! MATLAB fixture, read through the serde-native column-by-name surface.
//!
//! `tests/data/matlab/test_tables_v73.mat` is genuine real-MATLAB v7.3
//! output (BSD-3, `foreverallama/matio`; see `tests/data/matlab/NOTICE.md`).
//! A `table` / `timetable` decodes so each column is addressable by its MATLAB
//! variable name; a column that is itself an MCOS object (`string`, `datetime`,
//! `duration`, a struct, a user class) is resolved through its embedded
//! reference rather than surfacing as raw `uint32` metadata. Expected values are
//! transcribed from the generator script `tests/data/generators/test_tables_gen.m`
//! in that upstream project, so the oracle is independent of this crate's reader.
#![cfg(feature = "serde")]

use hdf5_pure::mat::{self, MatDatetime, MatDuration};
use serde::Deserialize;

fn read<T: serde::de::DeserializeOwned>() -> T {
    let bytes = std::fs::read("tests/data/matlab/test_tables_v73.mat")
        .expect("read test_tables_v73.mat fixture");
    mat::from_bytes(&bytes).expect("decode fixture")
}

#[test]
fn table_numeric_columns_decode() {
    // `table([1.1;2.2;3.3], [4.4;5.5;6.6])` — pure-numeric columns.
    #[derive(Deserialize)]
    struct File {
        table_numeric: Cols,
    }
    #[derive(Deserialize)]
    struct Cols {
        #[serde(rename = "Var1")]
        var1: Vec<f64>,
        #[serde(rename = "Var2")]
        var2: Vec<f64>,
    }
    let f: File = read();
    assert_eq!(f.table_numeric.var1, vec![1.1, 2.2, 3.3]);
    assert_eq!(f.table_numeric.var2, vec![4.4, 5.5, 6.6]);
}

#[test]
fn table_string_column_decodes() {
    // `table(["apple";"banana";"cherry"])` — a nested `string` column.
    #[derive(Deserialize)]
    struct File {
        table_strings: Cols,
    }
    #[derive(Deserialize)]
    struct Cols {
        #[serde(rename = "Var1")]
        var1: Vec<String>,
    }
    let f: File = read();
    assert_eq!(f.table_strings.var1, vec!["apple", "banana", "cherry"]);
}

#[test]
fn table_datetime_and_duration_columns_decode() {
    // `table(datetime(2020,1,1)+days(0:2)', seconds([30;60;90]))`.
    #[derive(Deserialize)]
    struct File {
        table_time: Cols,
    }
    #[derive(Deserialize)]
    struct Cols {
        #[serde(rename = "Time")]
        time: MatDatetime,
        #[serde(rename = "Duration")]
        duration: MatDuration,
    }
    let f: File = read();
    assert_eq!(
        f.table_time.time.millis_utc,
        vec![
            1_577_836_800_000.0,
            1_577_923_200_000.0,
            1_578_009_600_000.0
        ]
    );
    assert_eq!(
        f.table_time.duration.millis,
        vec![30_000.0, 60_000.0, 90_000.0]
    );
    assert_eq!(f.table_time.duration.seconds(), vec![30.0, 60.0, 90.0]);
}

#[test]
fn table_nan_and_missing_string_columns_decode() {
    // `table([1.1;NaN;3.3], ["A";"";"C"])`.
    #[derive(Deserialize)]
    struct File {
        table_nan: Cols,
    }
    #[derive(Deserialize)]
    struct Cols {
        #[serde(rename = "Var1")]
        var1: Vec<f64>,
        #[serde(rename = "Var2")]
        var2: Vec<String>,
    }
    let f: File = read();
    assert_eq!(f.table_nan.var1[0], 1.1);
    assert!(f.table_nan.var1[1].is_nan());
    assert_eq!(f.table_nan.var1[2], 3.3);
    assert_eq!(f.table_nan.var2, vec!["A", "", "C"]);
}

#[test]
fn table_heterogeneous_cell_column_decodes() {
    // `table({1; 'text'; datetime(2023,1,1)})` — a cell column mixing a double,
    // a char row, and a nested `datetime`.
    #[derive(Deserialize)]
    struct File {
        table_from_cell: Cols,
    }
    #[derive(Deserialize)]
    struct Cols {
        #[serde(rename = "Var1")]
        var1: (f64, String, MatDatetime),
    }
    let f: File = read();
    let (num, text, dt) = f.table_from_cell.var1;
    assert_eq!(num, 1.0);
    assert_eq!(text, "text");
    assert_eq!(dt.millis_utc, vec![1_672_531_200_000.0]);
}

#[test]
fn table_struct_and_user_object_columns_decode() {
    // `table({S;S;S}, {obj;obj;obj})` — a struct column and a user-class
    // (`TestClasses.BasicClass`) column; both decode losslessly as structs.
    #[derive(Deserialize)]
    struct File {
        table_with_objects: Cols,
    }
    #[derive(Deserialize)]
    struct Cols {
        #[serde(rename = "C")]
        c: Vec<SCell>,
        #[serde(rename = "Var2")]
        var2: Vec<BasicClass>,
    }
    #[derive(Deserialize)]
    struct SCell {
        field1: f64,
        field2: String,
    }
    #[derive(Deserialize)]
    struct BasicClass {
        a: f64,
    }
    let f: File = read();
    assert_eq!(f.table_with_objects.c.len(), 3);
    assert_eq!(f.table_with_objects.c[0].field1, 123.0);
    assert_eq!(f.table_with_objects.c[0].field2, "abc");
    assert_eq!(f.table_with_objects.var2.len(), 3);
    assert_eq!(f.table_with_objects.var2[0].a, 1.0);
}
