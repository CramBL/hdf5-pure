//! Native fixtures for edits to variable-length attributes.

use std::path::Path;

use hdf5::Location;
use hdf5::types::VarLenAscii;

use crate::absence;
use crate::file;

/// Creates a file containing a variable-length `label` attribute and a guard dataset.
///
/// The attribute initially contains `"before"`. Dense fixtures include eight numeric
/// attributes. The guard dataset contains 8,192 bytes of value `0x5a`.
///
/// # Panics
///
/// Panics if a file operation fails.
pub fn write(path: &Path, owner: Owner, storage: Storage) {
    let native = file::libhdf5_create_v18(path);
    match owner {
        Owner::Dataset => {
            native
                .new_dataset::<i32>()
                .shape([1])
                .create("d")
                .unwrap()
                .write(&[23])
                .unwrap();
        }
        Owner::Group => {
            native.create_group("g").unwrap();
        }
        Owner::Root => {}
    }
    let location = owner.location(&native);
    location
        .new_attr::<VarLenAscii>()
        .shape(())
        .create("label")
        .unwrap()
        .write_scalar(&VarLenAscii::from_ascii(b"before").unwrap())
        .unwrap();
    for index in 0..storage.extras() {
        location
            .new_attr::<i64>()
            .shape(())
            .create(format!("a{index}").as_str())
            .unwrap()
            .write_scalar(&i64::try_from(index).unwrap())
            .unwrap();
    }
    assert_eq!(location.attr_names().unwrap().len(), storage.extras() + 1);
    drop(location);
    native.flush().unwrap();
    native
        .new_dataset::<u8>()
        .shape([8_192])
        .create("guard")
        .unwrap()
        .write(&vec![0x5a; 8_192])
        .unwrap();
    native.close().unwrap();
}

/// Checks the edited attribute, untouched numeric attributes, and dataset values.
///
/// Uses the expectations for [`write()`] after applying `edit` to the `label` attribute.
///
/// # Panics
///
/// Panics if a file operation fails or a value differs from the expected contents.
pub fn assert_values(path: &Path, owner: Owner, storage: Storage, edit: Edit) {
    let native = file::libhdf5_open(path);
    let location = owner.location(&native);
    match edit {
        Edit::Numeric => assert_eq!(
            location
                .attr("label")
                .unwrap()
                .read_scalar::<i64>()
                .unwrap(),
            19
        ),
        Edit::Remove => {
            absence::assert_libhdf5_absent(&location.attr("label").unwrap_err(), "label")
        }
        Edit::Variable => assert_eq!(
            location
                .attr("label")
                .unwrap()
                .read_scalar::<VarLenAscii>()
                .unwrap()
                .as_str(),
            "after"
        ),
    }
    for index in 0..storage.extras() {
        assert_eq!(
            location
                .attr(&format!("a{index}"))
                .unwrap()
                .read_scalar::<i64>()
                .unwrap(),
            i64::try_from(index).unwrap()
        );
    }
    drop(location);
    assert_eq!(
        native.dataset("guard").unwrap().read_raw::<u8>().unwrap(),
        vec![0x5a; 8_192]
    );
    if matches!(owner, Owner::Dataset) {
        assert_eq!(
            native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
            [23]
        );
    }
    native.close().unwrap();
}

/// The object that owns the fixture's `label` attribute.
#[derive(Clone, Copy, Debug)]
pub enum Owner {
    /// A one-element integer dataset named `d`.
    Dataset,
    /// A group named `g`.
    Group,
    /// The root group.
    Root,
}

impl Owner {
    /// Returns the owner's path, with an empty string for the root group.
    pub fn path(self) -> &'static str {
        match self {
            Self::Dataset => "d",
            Self::Group => "g",
            Self::Root => "",
        }
    }

    fn location(self, file: &hdf5::File) -> Location {
        match self {
            Self::Dataset => (**file.dataset("d").unwrap()).clone(),
            Self::Group => (*file.group("g").unwrap()).clone(),
            Self::Root => (*file.group("/").unwrap()).clone(),
        }
    }
}

/// The initial storage of the fixture's attribute set.
#[derive(Clone, Copy, Debug)]
pub enum Storage {
    /// The `label` attribute alone in the object header.
    Compact,
    /// The `label` attribute and eight numeric attributes in dense storage.
    Dense,
}

impl Storage {
    fn extras(self) -> usize {
        match self {
            Self::Compact => 0,
            Self::Dense => 8,
        }
    }
}

/// The edit whose result [`assert_values`] checks.
#[derive(Clone, Copy, Debug)]
pub enum Edit {
    /// Replaces `label` with the integer 19.
    Numeric,
    /// Removes `label`.
    Remove,
    /// Replaces `label` with the variable-length string `"after"`.
    Variable,
}
