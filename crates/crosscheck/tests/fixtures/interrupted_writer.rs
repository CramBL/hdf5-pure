//! A libhdf5 writer process for tests that interrupt ordinary and SWMR sessions.

use std::env;
use std::error::Error as StdError;
use std::io::{self, Write};

use hdf5::file::LibraryVersion;

fn main() -> Result<(), Box<dyn StdError>> {
    let mut args = env::args_os().skip(1);
    let path = args.next().ok_or("missing fixture path")?;
    let mode = args.next().ok_or("missing writer mode")?;
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::V110))
        .create(path)?;
    file.new_dataset::<i32>()
        .shape((4,))
        .chunk((2,))
        .create("d")?
        .write(&[1, 2, 3, 4])?;
    match mode.to_str() {
        Some("ordinary") => {}
        Some("swmr") => file.start_swmr()?,
        _ => return Err("unknown writer mode".into()),
    }
    file.flush()?;
    println!("HDF5_PURE_WRITER_READY");
    io::stdout().flush()?;
    io::stdin().read_line(&mut String::new())?;
    Ok(())
}
