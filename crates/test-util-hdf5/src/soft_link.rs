//! Reference files with soft links in compact, dense, or symbol table groups.

use std::path::Path;

use hdf5::{Group, GroupStorageType};

use crate::file;

/// The group storage format used for generated link fixtures.
#[derive(Clone, Copy, Debug)]
pub enum GroupFormat {
    /// Link messages in group object headers.
    Compact,
    /// Link messages in fractal heaps with version 2 B-tree indexes.
    Dense,
    /// Version 1 symbol table groups.
    SymbolTable,
}

impl GroupFormat {
    /// Writes linked groups and datasets at `path` and checks their storage format.
    ///
    /// # Panics
    ///
    /// Panics if file creation or link creation fails, or a group uses a different format.
    pub fn generate(self, path: &Path) {
        let file = match self {
            Self::SymbolTable => file::libhdf5_create_earliest(path),
            Self::Compact | Self::Dense => file::libhdf5_create_v18(path),
        };
        let root = file.group("/").unwrap();
        let group = file.create_group("g").unwrap();
        let links = file.create_group("links").unwrap();
        let cycles = file.create_group("cycles").unwrap();
        let paths = file.create_group("paths").unwrap();
        write_value(&root, 99);
        write_value(&group, 42);
        write_value(&links, 7);
        write_value(&paths, 11);
        links.link_soft("/g/value", "absolute").unwrap();
        links.link_soft("value", "relative").unwrap();
        links.link_soft("relative", "chain").unwrap();
        links.link_soft("/g", "group").unwrap();
        links.link_soft("missing", "dangling").unwrap();
        links.link_soft("self", "self").unwrap();
        if !matches!(self, Self::SymbolTable) {
            links
                .link_external("absent.h5", "/value", "external")
                .unwrap();
        }
        cycles.link_soft("b", "a").unwrap();
        cycles.link_soft("a", "b").unwrap();
        cycles.link_soft("growing/suffix", "growing").unwrap();
        paths.link_soft(".", "current").unwrap();
        paths.link_soft("/", "root").unwrap();
        paths.link_hard("/g", "..").unwrap();
        paths.link_soft("..//./value/", "two_dots").unwrap();
        for group in [&root, &group, &links, &cycles, &paths] {
            if matches!(self, Self::Dense) {
                for index in 0..12 {
                    group
                        .link_soft("/g/value", &format!("filler{index:02}"))
                        .unwrap();
                }
            }
            assert_eq!(group.info().unwrap().storage_type, self.storage_type());
        }
        drop((root, group, links, cycles, paths));
        file.close().unwrap();
    }

    fn storage_type(self) -> GroupStorageType {
        match self {
            Self::Compact => GroupStorageType::Compact,
            Self::Dense => GroupStorageType::Dense,
            Self::SymbolTable => GroupStorageType::SymbolTable,
        }
    }
}

fn write_value(group: &Group, value: i32) {
    group
        .new_dataset::<i32>()
        .shape([1])
        .create("value")
        .unwrap()
        .write_raw(&[value])
        .unwrap();
}
