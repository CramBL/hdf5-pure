//! Editing an existing file in place with `File::open_rw`: add, overwrite, copy
//! (within the file and across two open files), and delete objects, and edit
//! group attributes, without rewriting the file from scratch.
//!
//! New data and rebuilt object headers are appended at end-of-file and the
//! superblock is repointed last, so a failed commit leaves the original file
//! valid and the cost is proportional to what changes.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example edit_in_place
//! ```

use hdf5_pure::DeflateLevel;
use hdf5_pure::{AttrValue, File, FileBuilder};

fn main() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("data.h5");

    // ---- Start from an existing file ------------------------------------
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("temperature")
        .with_f64_data(&[22.5, 23.1, 21.8]);
    let mut sensors = builder.create_group("sensors");
    sensors.create_dataset("pressure").with_f32_data(&[101.3]);
    builder.add_group(sensors.finish());
    builder.write(&path).expect("write initial file");
    let original_len = std::fs::metadata(&path).unwrap().len();

    // ---- A second, separate file to copy an object out of ---------------
    let lib_path = dir.path().join("library.h5");
    let mut lib_builder = FileBuilder::new();
    lib_builder
        .create_dataset("calibration")
        .with_f64_data(&[0.99, 1.00, 1.01]);
    lib_builder.write(&lib_path).expect("write library file");
    let library = File::open(&lib_path).expect("open library for reading");

    // ---- Edit it in place -----------------------------------------------
    let session = File::open_rw(&path).expect("open for editing");
    session
        .root()
        .create_group_with("run2", |g| {
            g.set_attr("kind", AttrValue::AsciiString("trial".into()));
        })
        .unwrap();
    session
        .root()
        .create_dataset("run2/signal", |b| {
            b.with_f64_data(&[1.0, 2.0, 3.0]);
        })
        .unwrap();
    // A chunked, shuffled, deflate-compressed dataset can be added in place too;
    // its chunk data and index are appended just like a contiguous blob.
    let waveform: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.01).sin()).collect();
    session
        .root()
        .create_dataset("run2/waveform", |b| {
            b.with_f64_data(&waveform)
                .with_chunks(&[512])
                .with_shuffle()
                .with_deflate(DeflateLevel::Six);
        })
        .unwrap();
    // Overwrite an existing dataset's values in place (H5Dwrite). Same datatype
    // and shape (3 f64), so the new bytes go straight into the existing block.
    session
        .dataset("temperature")
        .unwrap()
        .write_staged(|b| {
            b.with_f64_data(&[24.0, 24.4, 23.9]);
        })
        .unwrap();
    session.copy("temperature", "temperature_backup").unwrap(); // H5Ocopy (same file)
    // Cross-file H5Ocopy: pull an object out of another open file.
    session
        .copy_from(&library, "calibration", "run2/calibration")
        .expect("copy across files");
    session.root().delete("sensors/pressure").unwrap(); // H5Ldelete
    session.commit().expect("commit edits");
    drop(session); // release the editor's exclusive lock before reopening to read

    // ---- Verify ---------------------------------------------------------
    let file = File::open(&path).expect("reopen");
    let signal = file.dataset("run2/signal").unwrap().read_f64().unwrap();
    let waveform_read = file.dataset("run2/waveform").unwrap().read_f64().unwrap();
    let backup = file
        .dataset("temperature_backup")
        .unwrap()
        .read_f64()
        .unwrap();
    let run2_attrs = file.group("run2").unwrap().attrs().unwrap();
    let calibration = file
        .dataset("run2/calibration")
        .unwrap()
        .read_f64()
        .unwrap();

    let temperature = file.dataset("temperature").unwrap().read_f64().unwrap();

    println!("added   run2/signal        = {signal:?}");
    println!("overwr. temperature        = {temperature:?}");
    println!(
        "added   run2/waveform      = {} compressed samples",
        waveform_read.len()
    );
    println!("copied  temperature_backup = {backup:?}");
    println!("copied  run2/calibration   = {calibration:?} (from library.h5)");
    println!("group   run2.kind          = {:?}", run2_attrs.get("kind"));
    println!(
        "deleted sensors/pressure   -> {:?}",
        file.dataset("sensors/pressure").is_err()
    );
    println!(
        "\nfile grew from {original_len} to {} bytes",
        std::fs::metadata(&path).unwrap().len()
    );

    assert_eq!(signal, vec![1.0, 2.0, 3.0]);
    assert_eq!(waveform_read, waveform);
    assert_eq!(temperature, vec![24.0, 24.4, 23.9]); // overwritten value
    // The backup is a copy of the pre-commit on-disk dataset, captured before the
    // overwrite was applied, so it keeps the original values.
    assert_eq!(backup, vec![22.5, 23.1, 21.8]);
    assert_eq!(calibration, vec![0.99, 1.00, 1.01]);
    assert!(file.dataset("sensors/pressure").is_err());
    println!("in-place edits verified");
}
