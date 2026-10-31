//! Reference files with soft links in compact, dense, or symbol table groups.

#[cfg(feature = "__hdf5-1.10")]
use std::ffi::CString;
use std::path::Path;

#[cfg(feature = "__hdf5-1.10")]
use hdf5::LinkType;
use hdf5::{Group, GroupStorageType};
#[cfg(feature = "__hdf5-1.10")]
use hdf5_sys::h5l::H5Lget_val;
#[cfg(feature = "__hdf5-1.10")]
use hdf5_sys::h5p::H5P_DEFAULT;

#[cfg(feature = "__hdf5-1.10")]
use crate::__private::c_object;
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

/// A hard, soft, or external link, as libhdf5 reports it.
#[cfg(feature = "__hdf5-1.10")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeLink {
    /// A hard link to the object header at `address`.
    Hard { address: u64 },
    /// A soft link whose value is `target`.
    Soft { target: String },
    /// An external link to the object at `object_path` in the file `filename`.
    External {
        filename: String,
        object_path: String,
    },
}

/// Returns the name and the target of each link of `group`, in the order libhdf5 visits them.
///
/// # Panics
///
/// Panics if a libhdf5 call fails, or if the value of a soft or an external link is longer than
/// 1022 bytes or has a path that is not UTF-8.
#[cfg(feature = "__hdf5-1.10")]
pub fn native_links(file: &hdf5::File, group: &str) -> Vec<(String, NativeLink)> {
    let mut links = Vec::new();
    file.group(group)
        .unwrap()
        .iter_visit_default(|name, info| {
            links.push((String::from(name), info.link_type));
            Ok(())
        })
        .unwrap();
    links
        .into_iter()
        .map(|(name, link_type)| {
            let path = format!("{group}/{name}");
            let link = match link_type {
                LinkType::Hard => NativeLink::Hard {
                    address: c_object::header_address(file, &path),
                },
                LinkType::Soft => NativeLink::Soft {
                    target: String::from_utf8(link_value(file, &path)).unwrap(),
                },
                LinkType::External => external_link(&link_value(file, &path)),
            };
            (name, link)
        })
        .collect()
}

#[cfg(feature = "__hdf5-1.10")]
fn external_link(value: &[u8]) -> NativeLink {
    let (_version_and_flags, strings) = value.split_first().unwrap();
    let mut strings = strings
        .split(|&byte| byte == 0)
        .map(|string| String::from_utf8(string.to_vec()).unwrap());
    NativeLink::External {
        filename: strings.next().unwrap(),
        object_path: strings.next().unwrap(),
    }
}

#[cfg(feature = "__hdf5-1.10")]
fn link_value(file: &hdf5::File, path: &str) -> Vec<u8> {
    let name = CString::new(path).unwrap();
    let mut value = vec![0_u8; 1024];
    let status = hdf5::sync::sync(|| {
        // SAFETY: The library lock serializes access, `file` keeps the file open, `name` is a
        // NUL-terminated string that outlives the call, and `value` is writable for the length
        // passed beside it.
        unsafe {
            H5Lget_val(
                file.id(),
                name.as_ptr(),
                value.as_mut_ptr().cast(),
                value.len(),
                H5P_DEFAULT,
            )
        }
    });
    assert!(status >= 0, "H5Lget_val failed for {path:?}");
    let end = value
        .iter()
        .rposition(|&byte| byte != 0)
        .map_or(0, |last| last + 1);
    assert!(
        end + 1 < value.len(),
        "the value of {path:?} may be truncated to {end} bytes"
    );
    value.truncate(end);
    value
}
