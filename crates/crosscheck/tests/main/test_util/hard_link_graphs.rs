//! Native group fixtures with repeated hard links or tree-shaped object graphs.

#![cfg(feature = "__hdf5-1.10")]

use std::path::Path;

use test_util_hdf5::file;

/// A source group at `g` that reaches one object through two hard links.
#[derive(Clone, Copy, Debug)]
pub(crate) enum RepeatedObjectGraph {
    AncestorCycle,
    SelfLinked,
    SharedDataset,
    SharedSubgroup,
}

impl RepeatedObjectGraph {
    pub(crate) fn write(self, path: &Path) {
        let (file, g) = create_source_group(path);
        match self {
            Self::AncestorCycle => {
                g.create_group("child").unwrap();
                file.link_hard("g", "g/child/back").unwrap();
            }
            Self::SelfLinked => file.link_hard("g", "g/self").unwrap(),
            Self::SharedDataset => {
                write_i32(&g, "d1", &[1, 2, 3]);
                file.link_hard("g/d1", "g/d2").unwrap();
            }
            Self::SharedSubgroup => {
                write_i32(
                    &g.create_group("a").unwrap().create_group("sub").unwrap(),
                    "d",
                    &[4],
                );
                g.create_group("b").unwrap();
                file.link_hard("g/a/sub", "g/b/sub").unwrap();
            }
        }
        file.close().unwrap();
    }
}

/// A source group at `g` whose own hard links target each of its objects once.
#[derive(Clone, Copy, Debug)]
pub(crate) enum TreeGraph {
    /// A dataset of the source with a second hard link outside it.
    AliasOutsideSource,
    /// [`DEEP_CHAIN`] nested groups with a dataset in the innermost one.
    DeepChain,
    /// [`WIDE_FAN_OUT`] groups of [`WIDE_FAN_OUT`] groups of [`WIDE_FAN_OUT`] datasets each.
    Wide,
}

impl TreeGraph {
    pub(crate) fn write(self, path: &Path) {
        let (file, g) = create_source_group(path);
        match self {
            Self::AliasOutsideSource => {
                write_i32(&g, "d", &[5]);
                file.link_hard("g/d", "latest").unwrap();
            }
            Self::DeepChain => {
                let leaf = (0..DEEP_CHAIN).fold(g, |group, _| group.create_group("g").unwrap());
                write_i32(&leaf, "leaf", &[7]);
            }
            Self::Wide => {
                for outer in 0..WIDE_FAN_OUT {
                    let outer_group = g.create_group(&outer.to_string()).unwrap();
                    for inner in 0..WIDE_FAN_OUT {
                        let inner_group = outer_group.create_group(&inner.to_string()).unwrap();
                        for leaf in 0..WIDE_FAN_OUT {
                            write_i32(
                                &inner_group,
                                &leaf.to_string(),
                                &[wide_value(outer, inner, leaf)],
                            );
                        }
                    }
                }
            }
        }
        file.close().unwrap();
    }

    pub(crate) fn assert_read_by_c_library(self, group: &hdf5::Group) {
        match self {
            Self::AliasOutsideSource => {
                assert_eq!(group.member_names().unwrap(), vec!["d"]);
                assert_eq!(
                    group.dataset("d").unwrap().read_raw::<i32>().unwrap(),
                    vec![5]
                );
            }
            Self::DeepChain => {
                let leaf = (0..DEEP_CHAIN).fold(group.clone(), |group, _| {
                    assert_eq!(group.member_names().unwrap(), vec!["g"]);
                    group.group("g").unwrap()
                });
                assert_eq!(leaf.member_names().unwrap(), vec!["leaf"]);
                assert_eq!(
                    leaf.dataset("leaf").unwrap().read_raw::<i32>().unwrap(),
                    vec![7]
                );
            }
            Self::Wide => {
                for outer in 0..WIDE_FAN_OUT {
                    for inner in 0..WIDE_FAN_OUT {
                        for leaf in 0..WIDE_FAN_OUT {
                            let path = format!("{outer}/{inner}/{leaf}");
                            assert_eq!(
                                group.dataset(&path).unwrap().read_raw::<i32>().unwrap(),
                                vec![wide_value(outer, inner, leaf)],
                                "{path}"
                            );
                        }
                    }
                }
            }
        }
    }
}

fn create_source_group(path: &Path) -> (hdf5::File, hdf5::Group) {
    let file = file::libhdf5_create_v110(path);
    let g = file.create_group("g").unwrap();
    (file, g)
}

fn write_i32(group: &hdf5::Group, name: &str, values: &[i32]) {
    group
        .new_dataset::<i32>()
        .shape((values.len(),))
        .create(name)
        .unwrap()
        .write(values)
        .unwrap();
}

fn wide_value(outer: i32, inner: i32, leaf: i32) -> i32 {
    (outer * WIDE_FAN_OUT + inner) * WIDE_FAN_OUT + leaf
}

const DEEP_CHAIN: usize = 10_000;

const WIDE_FAN_OUT: i32 = 8;
