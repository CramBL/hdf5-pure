//! HDF5 fixture topologies shared by pure-reader and interoperability tests.

use std::path::Path;

#[cfg(feature = "hdf5")]
use hdf5::file::LibraryVersion;
use hdf5_pure::{AttrValue, File, FileBuilder};

/// Adds a `#refs#` group and two-dimensional row and column reference datasets.
pub fn path_references_2d(builder: &mut FileBuilder) {
    let mut refs = builder.create_group("#refs#");
    refs.create_dataset("a").with_f64_data(&[1.0]);
    refs.create_dataset("b").with_f64_data(&[2.0]);
    refs.create_dataset("c").with_f64_data(&[3.0]);
    refs.create_dataset("d").with_f64_data(&[4.0]);
    builder.add_group(refs.finish());

    builder
        .create_dataset("row_refs")
        .with_path_references(&["#refs#/a", "#refs#/b", "#refs#/c", "#refs#/d"])
        .with_shape(&[1, 4]);
    builder
        .create_dataset("col_refs")
        .with_path_references(&["#refs#/a", "#refs#/b"])
        .with_shape(&[2, 1]);
}

/// Adds the MATLAB v7.3 `#refs#` and `#subsystem#` reference graph.
pub fn matlab_refs_subsystem(builder: &mut FileBuilder) {
    let mut refs = builder.create_group("#refs#");
    refs.create_dataset("a")
        .with_u16_data(&[72, 101, 108, 108, 111]);
    refs.create_dataset("b")
        .with_u16_data(&[87, 111, 114, 108, 100]);
    refs.create_dataset("c").with_u16_data(&[70, 111, 111]);
    refs.create_dataset("cell_data")
        .with_path_references(&["#refs#/a", "#refs#/b", "#refs#/c"])
        .with_shape(&[1, 3]);
    refs.create_dataset("type_info")
        .with_u8_data(&[0, 0, 0, 0, 0, 0, 1, 0])
        .set_attr("MATLAB_class", AttrValue::AsciiString("string".into()));
    builder.add_group(refs.finish());

    let mut subsystem = builder.create_group("#subsystem#");
    subsystem
        .create_dataset("MCOS")
        .with_path_references(&["#refs#/cell_data", "#refs#/type_info"])
        .with_shape(&[2, 1]);
    builder.add_group(subsystem.finish());

    builder
        .create_dataset("data")
        .with_path_references(&["#refs#/a"])
        .with_shape(&[1, 1])
        .set_attr("MATLAB_class", AttrValue::AsciiString("string".into()));
}

/// Adds a MATLAB-style struct group with field metadata and two numeric datasets.
pub fn nested_matlab_struct(builder: &mut FileBuilder) {
    let mut group = builder.create_group("my_struct");
    group.create_dataset("x").with_f64_data(&[1.0, 2.0]);
    group.create_dataset("y").with_f64_data(&[3.0, 4.0]);
    group.set_attr("MATLAB_class", AttrValue::AsciiString("struct".into()));
    group.set_attr(
        "MATLAB_fields",
        AttrValue::VarLenAsciiCharArray(vec!["x".into(), "y".into()]),
    );
    builder.add_group(group.finish());
}

/// Adds a MATLAB-style struct whose direct children are groups rather than datasets.
pub fn group_only_matlab_struct(builder: &mut FileBuilder) {
    let mut outer = builder.create_group("outer");
    outer.set_attr("MATLAB_class", AttrValue::AsciiString("struct".into()));

    let mut child_a = outer.create_group("a");
    child_a.create_dataset("val").with_f64_data(&[1.0]);
    child_a.set_attr("MATLAB_class", AttrValue::AsciiString("double".into()));
    outer.add_group(child_a.finish());

    let mut child_b = outer.create_group("b");
    child_b.create_dataset("val").with_i32_data(&[42]);
    child_b.set_attr("MATLAB_class", AttrValue::AsciiString("int32".into()));
    outer.add_group(child_b.finish());

    builder.add_group(outer.finish());
}

/// Adds the common object-reference target group `g/inner` to a pure-Rust builder.
pub fn reference_target_group(builder: &mut FileBuilder) {
    let mut group = builder.create_group("g");
    group.create_dataset("inner").with_i32_data(&[1, 2, 3]);
    builder.add_group(group.finish());
}

/// Adds the common object-reference target group `g/inner` to an open libhdf5 file.
#[cfg(feature = "hdf5")]
pub fn populate_libhdf5_reference_target(file: &hdf5::File) {
    let group = file.create_group("g").unwrap();
    group
        .new_dataset::<i32>()
        .shape((3,))
        .create("inner")
        .unwrap()
        .write(&[1i32, 2, 3])
        .unwrap();
}

/// Commits unrelated small datasets so recently freed object-header space is reused.
pub fn churn_commits(path: &Path, count: i32) {
    for i in 0..count {
        let session = File::open_rw(path).unwrap();
        session
            .root()
            .create_dataset(&format!("churn{i}"), |builder| {
                builder.with_i32_data(&[i]);
            })
            .unwrap();
        session.commit().unwrap();
    }
}

/// Moves `g` by adding `g/extra`, then churns the freed header space.
pub fn move_reference_target_and_churn(path: &Path, count: i32) {
    let session = File::open_rw(path).unwrap();
    session
        .root()
        .create_dataset("g/extra", |builder| {
            builder.with_i32_data(&[9]);
        })
        .unwrap();
    session.commit().unwrap();
    drop(session);
    churn_commits(path, count);
}

/// Adds the common edit/repack starter topology to a pure-Rust file builder.
///
/// The fixture contains `alpha`, `doomed`, and `grp/beta`; callers choose the
/// payload of `doomed` so tests can distinguish edit and repack scenarios.
pub fn edit_repack_starter(builder: &mut FileBuilder, doomed: &[i32]) {
    builder
        .create_dataset("alpha")
        .with_f64_data(&[1.0, 2.0, 3.0]);
    builder.create_dataset("doomed").with_i32_data(doomed);
    let mut group = builder.create_group("grp");
    group
        .create_dataset("beta")
        .with_i32_data(&[10, 20, 30, 40]);
    builder.add_group(group.finish());
}

/// Populates an open libhdf5 file with the common edit/repack starter topology.
#[cfg(feature = "hdf5")]
pub fn populate_libhdf5_edit_repack_starter(file: &hdf5::File, doomed: &[i32]) {
    file.new_dataset::<f64>()
        .shape((3,))
        .create("alpha")
        .unwrap()
        .write(&[1.0f64, 2.0, 3.0])
        .unwrap();
    file.new_dataset::<i32>()
        .shape((doomed.len(),))
        .create("doomed")
        .unwrap()
        .write(doomed)
        .unwrap();
    let group = file.create_group("grp").unwrap();
    group
        .new_dataset::<i32>()
        .shape((4,))
        .create("beta")
        .unwrap()
        .write(&[10i32, 20, 30, 40])
        .unwrap();
}

/// Creates a bounded libhdf5 file containing the common edit/repack starter topology.
///
/// The returned file remains open so a specialized fixture can add metadata before closing it.
#[cfg(feature = "hdf5")]
pub fn libhdf5_edit_repack_starter(
    path: &Path,
    low: LibraryVersion,
    high: LibraryVersion,
    doomed: &[i32],
) -> hdf5::File {
    let file = hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_bounds(low, high))
        .create(path)
        .unwrap();
    populate_libhdf5_edit_repack_starter(&file, doomed);
    file
}
