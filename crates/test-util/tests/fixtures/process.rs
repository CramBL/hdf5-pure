//! Explicit child operations for fixture lifecycle tests.

use std::env;
use std::error::Error as StdError;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn StdError>> {
    let operation = Operation::parse(env::args().nth(1).as_deref())?;
    match operation {
        Operation::Ready => {
            println!("READY");
            io::stdout().flush()?;
            io::stdin().read_line(&mut String::new())?;
        }
        Operation::Wrong => {
            eprintln!("fixture not ready");
            println!("WRONG");
            io::stdout().flush()?;
            io::stdin().read_line(&mut String::new())?;
        }
        Operation::Silent => {
            io::stdin().read_line(&mut String::new())?;
        }
        Operation::Output => {
            io::stdout().write_all(&vec![b'o'; 256 * 1024])?;
            io::stderr().write_all(&vec![b'e'; 256 * 1024])?;
        }
        Operation::Failure => {
            eprintln!("fixture failed");
            std::process::exit(17);
        }
        Operation::Fragmented => {
            io::stdout().write_all(b"RE")?;
            io::stdout().flush()?;
            std::thread::sleep(std::time::Duration::from_millis(20));
            io::stdout().write_all(b"ADY\n")?;
            io::stdout().flush()?;
            io::stdin().read_line(&mut String::new())?;
        }
        Operation::Exit => {}
        Operation::Growing => {
            eprintln!("fixture keeps writing");
            let mut stdout = io::stdout().lock();
            loop {
                stdout.write_all(&[b'o'; 1024])?;
            }
        }
    }
    Ok(())
}

enum Operation {
    Exit,
    Failure,
    Fragmented,
    Growing,
    Output,
    Ready,
    Silent,
    Wrong,
}

impl Operation {
    fn parse(value: Option<&str>) -> Result<Self, &'static str> {
        Ok(match value {
            Some("exit") => Self::Exit,
            Some("failure") => Self::Failure,
            Some("fragmented") => Self::Fragmented,
            Some("growing") => Self::Growing,
            Some("output") => Self::Output,
            Some("ready") => Self::Ready,
            Some("silent") => Self::Silent,
            Some("wrong") => Self::Wrong,
            _ => return Err("unknown process fixture operation"),
        })
    }
}
