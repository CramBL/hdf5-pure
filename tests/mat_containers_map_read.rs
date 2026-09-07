//! Reading MATLAB `containers.Map` variables straight into Rust maps, over a
//! real MATLAB fixture (BSD-3, `foreverallama/matio`; see
//! `tests/data/matlab/NOTICE.md`). Expected values come from the generator
//! `tests/data/generators/test_maps_gen.m` and the `test_containermap.py` oracle.
#![cfg(feature = "serde")]

use hdf5_pure::mat;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};

fn read<T: serde::de::DeserializeOwned>() -> T {
    let bytes = std::fs::read("tests/data/matlab/test_maps_v73.mat")
        .expect("read test_maps_v73.mat fixture");
    mat::from_bytes(&bytes).expect("decode fixture")
}

#[test]
fn char_keyed_map_decodes_to_string_keyed_map() {
    // `containers.Map({'a','b'}, [1,2])` -> {"a": 1.0, "b": 2.0}.
    #[derive(Deserialize)]
    struct File {
        map_char_keys: HashMap<String, f64>,
    }
    let m = read::<File>().map_char_keys;
    assert_eq!(m.len(), 2);
    assert_eq!(m["a"], 1.0);
    assert_eq!(m["b"], 2.0);
}

#[test]
fn string_keyed_map_matches_char_keyed_map() {
    // `containers.Map(["a","b"], [1,2])` decodes identically: MATLAB stores
    // `string` keys as `char` internally.
    #[derive(Deserialize)]
    struct File {
        map_string_keys: BTreeMap<String, f64>,
    }
    let m = read::<File>().map_string_keys;
    assert_eq!(
        m.into_iter().collect::<Vec<_>>(),
        [("a".to_owned(), 1.0), ("b".to_owned(), 2.0)]
    );
}

#[test]
fn numeric_keyed_map_stringifies_keys() {
    // `containers.Map([1 2], {'a' 'b'})`: numeric keys are presented as strings
    // (`1.0 -> "1"`), char values as `String`.
    #[derive(Deserialize)]
    struct File {
        map_numeric_keys: BTreeMap<String, String>,
    }
    let m = read::<File>().map_numeric_keys;
    assert_eq!(
        m.into_iter().collect::<Vec<_>>(),
        [
            ("1".to_owned(), "a".to_owned()),
            ("2".to_owned(), "b".to_owned())
        ]
    );
}

#[test]
fn empty_map_decodes_to_empty_map() {
    #[derive(Deserialize)]
    struct File {
        map_empty: HashMap<String, f64>,
    }
    assert!(read::<File>().map_empty.is_empty());
}

#[test]
fn map_deserializes_into_a_struct_keyed_by_field_name() {
    // The killer ergonomic: each map key becomes a struct field, so a
    // `deny_unknown_fields` struct of exactly the keys deserializes with no
    // internal MATLAB metadata leaking in.
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Lookup {
        a: f64,
        b: f64,
    }
    #[derive(Deserialize)]
    struct File {
        map_char_keys: Lookup,
    }
    let l = read::<File>().map_char_keys;
    assert_eq!((l.a, l.b), (1.0, 2.0));
}

#[test]
fn dictionary_still_reads_losslessly_for_now() {
    // `dictionary` is not yet given a typed decode; it must still read (not
    // error) via the lossless raw property map, exposing its `data` struct with
    // parallel `Key` / `Value`. This pins the deferred-but-graceful behavior.
    #[derive(Deserialize)]
    struct Data {
        #[serde(rename = "Key")]
        key: Vec<f64>,
        #[serde(rename = "Value")]
        value: Vec<String>,
    }
    #[derive(Deserialize)]
    struct Dict {
        data: Data,
    }
    #[derive(Deserialize)]
    struct File {
        dict_numeric_keys: Dict,
    }
    let d = read::<File>().dict_numeric_keys;
    assert_eq!(d.data.key, [1.0, 2.0, 3.0]);
    assert_eq!(d.data.value, ["apple", "banana", "cherry"]);
}
