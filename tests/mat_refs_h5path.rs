#![cfg(feature = "serde")]
//! `H5PATH` on the objects interned under `#refs#`, measured against MATLAB.
//!
//! MATLAB stamps every object it interns under `#refs#` with its own absolute
//! path. Nothing in this crate reads the attribute and no third-party reader is
//! known to need it — a reference resolves through HDF5 without it — but MATLAB
//! writes it on essentially every referenced object, and a `.mat` this crate
//! writes should not be the one that doesn't.
//!
//! [`matlab_stamps_every_refs_object_with_its_own_path`] re-derives the rule
//! from MATLAB's own files on every run rather than asserting it as a constant,
//! including both of its exceptions.

use hdf5_pure::mat::{self, EmptySequencePolicy, Options, StringClass};
use hdf5_pure::{AttrValue, File};
use serde::Serialize;

/// Every object in `file`, as `(path, is_group, H5PATH value)`.
fn objects(file: &File) -> Vec<(String, bool, Option<String>)> {
    fn h5path(attrs: &std::collections::HashMap<String, AttrValue>) -> Option<String> {
        match attrs.get("H5PATH") {
            Some(AttrValue::AsciiString(s)) | Some(AttrValue::String(s)) => Some(s.clone()),
            Some(other) => Some(format!("unexpected type: {other:?}")),
            None => None,
        }
    }
    fn walk(file: &File, path: &str, out: &mut Vec<(String, bool, Option<String>)>) {
        let group = match if path.is_empty() {
            Ok(file.root())
        } else {
            file.group(path)
        } {
            Ok(g) => g,
            Err(_) => return,
        };
        if !path.is_empty()
            && let Ok(attrs) = group.attrs()
        {
            out.push((path.to_string(), true, h5path(&attrs)));
        }
        for name in group.datasets().unwrap_or_default() {
            let full = if path.is_empty() {
                name.clone()
            } else {
                format!("{path}/{name}")
            };
            if let Ok(ds) = file.dataset(&full)
                && let Ok(attrs) = ds.attrs()
            {
                out.push((full, false, h5path(&attrs)));
            }
        }
        for sub in group.groups().unwrap_or_default() {
            let full = if path.is_empty() {
                sub.clone()
            } else {
                format!("{path}/{sub}")
            };
            walk(file, &full, out);
        }
    }
    let mut out = Vec::new();
    walk(file, "", &mut out);
    out
}

/// True for `#refs#/<name>` and nothing deeper.
fn is_refs_member(path: &str) -> bool {
    path.strip_prefix("#refs#/")
        .is_some_and(|rest| !rest.contains('/'))
}

fn matlab_class(file: &File, path: &str) -> Option<String> {
    let attrs = file.dataset(path).ok()?.attrs().ok()?;
    match attrs.get("MATLAB_class") {
        Some(AttrValue::AsciiString(s)) | Some(AttrValue::String(s)) => Some(s.clone()),
        _ => None,
    }
}

/// The measurement the writer's rule is built on, re-derived from MATLAB's own
/// files. The count assertion keeps a walk that found nothing from passing.
#[test]
fn matlab_stamps_every_refs_object_with_its_own_path() {
    let mut members = 0;
    let mut exceptions = Vec::new();
    for entry in std::fs::read_dir("tests/data/matlab").expect("the fixture directory") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("mat") {
            continue;
        }
        // Two fixtures are deliberately corrupted read-failure cases.
        let Ok(file) = File::open(&path) else {
            continue;
        };
        for (obj, _, h5path) in objects(&file) {
            if !is_refs_member(&obj) {
                continue;
            }
            members += 1;
            match h5path {
                Some(v) => assert_eq!(
                    v,
                    format!("/{obj}"),
                    "{}: {obj} carries an H5PATH that is not its own path",
                    path.display()
                ),
                None => exceptions.push((path.clone(), obj)),
            }
        }
    }

    assert!(
        members > 300,
        "expected hundreds of #refs# members across the fixtures, found {members} — \
         the walk is not reaching them"
    );

    // The only object MATLAB leaves unstamped is the canonical-empty
    // placeholder, and it does so in every file that has one.
    for (file, obj) in &exceptions {
        let f = File::open(file).unwrap();
        assert_eq!(
            matlab_class(&f, obj).as_deref(),
            Some("canonical empty"),
            "{}: {obj} has no H5PATH and is not the canonical empty",
            file.display()
        );
    }
    assert_eq!(
        exceptions.len(),
        8,
        "expected one canonical empty per readable fixture, got {exceptions:?}"
    );
}

/// Our writers follow that rule: every object interned under `#refs#` carries
/// its own path, the canonical empty carries nothing, and no object outside
/// `#refs#` carries the attribute at all.
///
/// Both emitters, because they intern `#refs#` objects through different code:
/// `to_bytes` walks the value tree straight into a `FileBuilder`, and
/// `to_bytes_with_options` goes through `MatBuilder`. Stamping one and not the
/// other is the exact drift this crate keeps finding, and a check through only
/// `to_bytes_with_options` is blind to the default path every caller of
/// `to_bytes` gets.
#[test]
fn our_refs_objects_carry_their_own_path() {
    #[derive(Serialize)]
    struct Element {
        label: String,
        weight: f64,
    }
    #[derive(Serialize)]
    struct Doc {
        // A cell of structs: struct *groups* under `#refs#`.
        elements: Vec<Element>,
        // Ragged, so it stays a cell of datasets rather than unifying.
        ragged: Vec<Vec<i32>>,
        // A `None` slot interns a `struct([])` marker of its own.
        optional: Vec<Option<f64>>,
        // An empty cell, which is written at the root rather than interned.
        empty: Vec<Vec<i32>>,
        // Drives the MCOS subsystem, whose helper refs follow the same rule.
        text: String,
    }
    let doc = Doc {
        elements: vec![
            Element {
                label: "a".into(),
                weight: 1.0,
            },
            Element {
                label: "b".into(),
                weight: 2.0,
            },
        ],
        ragged: vec![vec![1], vec![2, 3]],
        optional: vec![Some(1.0), None],
        empty: Vec::new(),
        text: "hello".into(),
    };
    let mut opts = Options::default();
    opts.empty_sequence_policy = EmptySequencePolicy::Cell;
    // The `string` class is what builds the MCOS subsystem, so this covers the
    // helper refs (payload, FileWrapper metadata, templates, alias) too.
    opts.string_class = StringClass::String;

    // The default-options document drops the two fields that need a non-default
    // option, so `to_bytes` still exercises interned structs and cells.
    #[derive(Serialize)]
    struct Plain {
        elements: Vec<Element>,
        ragged: Vec<Vec<i32>>,
        optional: Vec<Option<f64>>,
    }
    let plain = Plain {
        elements: vec![
            Element {
                label: "a".into(),
                weight: 1.0,
            },
            Element {
                label: "b".into(),
                weight: 2.0,
            },
        ],
        ragged: vec![vec![1], vec![2, 3]],
        optional: vec![Some(1.0), None],
    };

    for (label, bytes, min_members, expect_canonical) in [
        ("to_bytes", mat::to_bytes(&plain).unwrap(), 6, false),
        (
            "to_bytes_with_options",
            mat::to_bytes_with_options(&doc, &opts).unwrap(),
            12,
            true,
        ),
    ] {
        let file = File::from_bytes(bytes).unwrap();
        let objects = objects(&file);

        let members: Vec<_> = objects
            .iter()
            .filter(|(path, _, _)| is_refs_member(path))
            .collect();
        assert!(
            members.len() >= min_members,
            "{label}: expected at least {min_members} interned objects, got {}: {members:?}",
            members.len()
        );
        assert!(
            members.iter().any(|(_, is_group, _)| *is_group),
            "{label}: expected at least one struct group under #refs#, got {members:?}"
        );

        let mut unstamped = Vec::new();
        for (path, _, h5path) in &members {
            match h5path {
                Some(v) => assert_eq!(v, &format!("/{path}"), "{label}: {path}: wrong H5PATH"),
                None => unstamped.push(path.clone()),
            }
        }
        // Only the MCOS subsystem brings a canonical empty, and only the
        // `string` class builds a subsystem.
        assert_eq!(
            unstamped.len(),
            usize::from(expect_canonical),
            "{label}: unexpected unstamped objects: {unstamped:?}"
        );
        if expect_canonical {
            assert_eq!(
                matlab_class(&file, &unstamped[0]).as_deref(),
                Some("canonical empty"),
                "{label}: the unstamped object is the canonical empty"
            );
        }

        // Everything else — root variables, struct fields, `#subsystem#/MCOS`,
        // and the fields inside an interned struct — carries no H5PATH. MATLAB
        // stamps some of those, but with a value that is not their path (see
        // `mat::builder::refs_h5path`), and a wrong path is worse than none.
        let stray: Vec<_> = objects
            .iter()
            .filter(|(path, _, h5path)| h5path.is_some() && !is_refs_member(path))
            .collect();
        assert!(
            stray.is_empty(),
            "{label}: H5PATH outside #refs#'s immediate children: {stray:?}"
        );
    }
}
