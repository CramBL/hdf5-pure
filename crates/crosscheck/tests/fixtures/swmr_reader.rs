//! A native reader process for SWMR and normal file observations.

use std::env;
use std::error::Error as StdError;
use std::io::{self, Write};

use protocol::{Mode, Observation};

#[path = "swmr_reader/protocol.rs"]
mod protocol;

fn main() -> Result<(), Box<dyn StdError>> {
    let mut args = env::args_os().skip(1);
    let path = args.next().ok_or("missing fixture path")?;
    let mode = args.next().ok_or("missing reader mode")?;
    let mode: Mode = serde_json::from_str(mode.to_str().ok_or("mode is not Unicode")?)?;
    if args.next().is_some() {
        return Err("unexpected fixture argument".into());
    }
    io::stdout().write_all(protocol::START)?;
    io::stdout().flush()?;
    let file = hdf5::File::open_as(
        path,
        match mode {
            Mode::Normal => hdf5::OpenMode::Read,
            Mode::Swmr => hdf5::OpenMode::ReadSWMR,
        },
    )?;
    let dataset = file.dataset("d")?;
    if mode == Mode::Swmr {
        dataset.refresh()?;
    }
    let observation = Observation {
        mode,
        shape: dataset
            .shape()
            .try_into()
            .map_err(|shape: Vec<usize>| format!("expected rank-one dataset, got {shape:?}"))?,
        values: dataset.read_raw::<i32>()?,
    };
    drop(dataset);
    file.close()?;
    serde_json::to_writer(io::stdout().lock(), &observation)?;
    io::stdout().flush()?;
    Ok(())
}
