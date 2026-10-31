//! Demonstrates free-space reuse across editing sessions.
//!
//! Deletes a dataset and writes its replacement after reopening the file.
//! Compares persistent [`FsmAggr`](FileSpaceStrategy::FsmAggr) and [`Page`](FileSpaceStrategy::Page)
//! with nonpersistent [`FsmAggr`](FileSpaceStrategy::FsmAggr).
//!
//! Run with:
//!
//! ```bash
//! cargo run --example file_space
//! ```

use std::error::Error as StdError;
use std::path::Path;

use hdf5_pure::{File, FileBuilder, FileSpaceSettings, FileSpaceStrategy};

fn main() -> Result<(), Box<dyn StdError>> {
    let dir = tempfile::tempdir()?;
    for (filename, strategy, persist) in [
        ("managed.h5", FileSpaceStrategy::FsmAggr, true),
        ("paged.h5", FileSpaceStrategy::Page, true),
        ("control.h5", FileSpaceStrategy::FsmAggr, false),
    ] {
        println!("\n{strategy:?}, persistence = {persist}");
        replace_dataset_across_sessions(
            &dir.path().join(filename),
            FileSpaceSettings::new(strategy).with_persistence(persist),
        )?;
    }
    Ok(())
}

fn replace_dataset_across_sessions(
    path: &Path,
    settings: FileSpaceSettings,
) -> Result<(), Box<dyn StdError>> {
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(settings);
    builder
        .create_dataset("scratch")
        .with_f64_data(&vec![0.0; 4096]);
    builder.write(path)?;

    let session = File::open_rw(path)?;
    session.root().create_dataset("keep", |dataset| {
        dataset.with_i32_data(&vec![1; 4096]);
    })?;
    session.commit()?;
    drop(session);
    println!(
        "  before deletion: {} bytes",
        std::fs::metadata(path)?.len()
    );

    let session = File::open_rw(path)?;
    session.root().delete("scratch")?;
    session.commit()?;
    drop(session);

    let file = File::open(path)?;
    match file.persisted_free_sections()? {
        Some(report) => println!(
            "  after reopening: {} bytes of persisted free space",
            report.free_sections().free_bytes()
        ),
        None => println!("  after reopening: free space is not persisted"),
    }
    drop(file);

    let session = File::open_rw(path)?;
    session.root().create_dataset("replacement", |dataset| {
        dataset.with_f64_data(&vec![7.0; 4096]);
    })?;
    session.commit()?;
    drop(session);
    println!(
        "  after replacement: {} bytes",
        std::fs::metadata(path)?.len()
    );

    let file = File::open(path)?;
    if file.dataset("keep")?.read_i32()? != vec![1; 4096] {
        return Err("surviving dataset differs from the original data".into());
    }
    if file.dataset("replacement")?.read_f64()? != vec![7.0; 4096] {
        return Err("replacement dataset differs from the new data".into());
    }
    Ok(())
}
