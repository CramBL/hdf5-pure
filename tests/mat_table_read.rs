//! Typed `MatTable` / `MatTimetable` / `MatColumn` views over a real MATLAB
//! fixture (BSD-3, `foreverallama/matio`; see `tests/data/matlab/NOTICE.md`).
//! Expected values come from the generator `tests/data/generators/test_tables_gen.m`.
#![cfg(feature = "serde")]

use hdf5_pure::mat::{self, MatColumn, MatTable, MatTimetable};
use serde::Deserialize;

fn read<T: serde::de::DeserializeOwned>() -> T {
    let bytes = std::fs::read("tests/data/matlab/test_tables_v73.mat")
        .expect("read test_tables_v73.mat fixture");
    mat::from_bytes(&bytes).expect("decode fixture")
}

#[test]
fn mattable_shape_names_and_numeric_columns() {
    #[derive(Deserialize)]
    struct File {
        table_numeric: MatTable,
    }
    let t = read::<File>().table_numeric;
    assert_eq!(t.num_rows(), 3);
    assert_eq!(t.num_variables(), 2);
    assert_eq!(t.variable_names().collect::<Vec<_>>(), ["Var1", "Var2"]);
    assert!(t.row_names().is_empty());
    assert_eq!(t.column("Var1").unwrap().as_f64().unwrap(), [1.1, 2.2, 3.3]);
    assert_eq!(t.column("Var2").unwrap().as_f64().unwrap(), [4.4, 5.5, 6.6]);
    assert!(t.column("nope").is_none());
}

#[test]
fn mattable_text_datetime_duration_columns() {
    #[derive(Deserialize)]
    struct File {
        table_strings: MatTable,
        table_time: MatTable,
    }
    let f: File = read();

    assert_eq!(
        f.table_strings
            .column("Var1")
            .unwrap()
            .as_strings()
            .unwrap(),
        ["apple", "banana", "cherry"]
    );

    let t = f.table_time;
    match t.column("Time") {
        Some(MatColumn::Datetime(d)) => assert_eq!(
            d.millis_utc,
            vec![
                1_577_836_800_000.0,
                1_577_923_200_000.0,
                1_578_009_600_000.0
            ]
        ),
        other => panic!(
            "Time should be Datetime, got {:?}",
            other.map(MatColumn::kind)
        ),
    }
    match t.column("Duration") {
        Some(MatColumn::Duration(d)) => {
            assert_eq!(d.millis, vec![30_000.0, 60_000.0, 90_000.0]);
        }
        other => panic!(
            "Duration should be Duration, got {:?}",
            other.map(MatColumn::kind)
        ),
    }
}

#[test]
fn mattable_row_names_are_exposed() {
    #[derive(Deserialize)]
    struct File {
        table_row_names: MatTable,
    }
    let t = read::<File>().table_row_names;
    assert_eq!(t.row_names(), ["R1", "R2", "R3"]);
    assert_eq!(t.variable_names().collect::<Vec<_>>(), ["Col1", "Col2"]);
    assert_eq!(t.column("Col1").unwrap().as_f64().unwrap(), [1.0, 2.0, 3.0]);
}

#[test]
fn mattable_multicolumn_variable_keeps_shape() {
    // `multicoldata = [1,4;2,5;3,6]` is a single 3×2 variable in the table.
    #[derive(Deserialize)]
    struct File {
        table_multi_col_data: MatTable,
    }
    let t = read::<File>().table_multi_col_data;
    let m = t.column("multicoldata").unwrap().as_matrix().unwrap();
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 2);
    assert_eq!(m.data(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn mattable_struct_column_is_other() {
    #[derive(Deserialize)]
    struct File {
        table_with_objects: MatTable,
    }
    let t = read::<File>().table_with_objects;
    assert_eq!(t.column("C").unwrap().kind(), "other");
}

#[test]
fn mattimetable_datetime_rowtimes() {
    #[derive(Deserialize)]
    struct File {
        timetable_datetime: MatTimetable,
    }
    let tt = read::<File>().timetable_datetime;
    assert_eq!(tt.num_rows(), 3);
    assert_eq!(tt.variable_names().collect::<Vec<_>>(), ["data1"]);
    assert_eq!(
        tt.column("data1").unwrap().as_f64().unwrap(),
        [1.0, 2.0, 3.0]
    );
    match tt.row_times() {
        Some(MatColumn::Datetime(d)) => assert_eq!(
            d.millis_utc,
            vec![
                1_672_531_200_000.0,
                1_672_617_600_000.0,
                1_672_704_000_000.0
            ]
        ),
        other => panic!(
            "rowTimes should be Datetime, got {:?}",
            other.map(MatColumn::kind)
        ),
    }
}

#[test]
fn table_meta_key_does_not_leak_into_plain_targets() {
    // The reserved metadata entry must be invisible to ordinary targets: a
    // `deny_unknown_fields` struct of just the columns deserializes, and a
    // homogeneous column map sees only the real columns.
    use std::collections::BTreeMap;

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Strict {
        #[serde(rename = "Var1")]
        var1: Vec<f64>,
        #[serde(rename = "Var2")]
        var2: Vec<f64>,
    }
    #[derive(Deserialize)]
    struct File {
        table_numeric: Strict,
    }
    let f: File = read();
    assert_eq!(f.table_numeric.var1, vec![1.1, 2.2, 3.3]);
    assert_eq!(f.table_numeric.var2, vec![4.4, 5.5, 6.6]);

    #[derive(Deserialize)]
    struct MapFile {
        table_numeric: BTreeMap<String, Vec<f64>>,
    }
    let m: MapFile = read();
    assert_eq!(m.table_numeric.keys().collect::<Vec<_>>(), ["Var1", "Var2"]);
    assert_eq!(m.table_numeric["Var2"], vec![4.4, 5.5, 6.6]);
}

#[test]
fn mattimetable_duration_rowtimes() {
    #[derive(Deserialize)]
    struct File {
        timetable_duration: MatTimetable,
    }
    let tt = read::<File>().timetable_duration;
    match tt.row_times() {
        Some(MatColumn::Duration(d)) => {
            assert_eq!(d.millis, vec![10_000.0, 20_000.0, 30_000.0]);
        }
        other => panic!(
            "rowTimes should be Duration, got {:?}",
            other.map(MatColumn::kind)
        ),
    }
}
