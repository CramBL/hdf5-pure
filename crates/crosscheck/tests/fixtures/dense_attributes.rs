//! Native attribute reads and writes for dense-storage interoperability tests.

use std::env;
use std::error::Error as StdError;
use std::io::{self, Write};
use std::path::Path;

use test_util_hdf5::absence;

use protocol::{Attribute, Frame, Lookup, Operation, Rejection, Response, Stage};

#[path = "dense_attributes/protocol.rs"]
mod protocol;

fn main() -> Result<(), Box<dyn StdError>> {
    let mut args = env::args_os().skip(1);
    let path = args.next().ok_or("missing fixture path")?;
    let operation = args.next().ok_or("missing fixture operation")?;
    let operation: Operation =
        serde_json::from_str(operation.to_str().ok_or("operation is not Unicode")?)?;
    if args.next().is_some() {
        return Err("unexpected fixture argument".into());
    }
    io::stdout().write_all(protocol::START)?;
    io::stdout().flush()?;
    let response = match execute(Path::new(&path), operation) {
        Ok(response) => response,
        Err(Failure::Native(rejection)) => Response::Rejected(rejection),
        Err(Failure::Infrastructure(error)) => return Err(error),
    };
    serde_json::to_writer(io::stdout().lock(), &response)?;
    Ok(())
}

fn execute(path: &Path, operation: Operation) -> Result<Response, Failure> {
    if matches!(operation, Operation::InsertHuge | Operation::InsertManaged) {
        let len = if matches!(operation, Operation::InsertHuge) {
            12_000
        } else {
            16
        };
        let file = native(Stage::Open, hdf5::File::open_rw(path))?;
        let added: Vec<i64> = (0..len).collect();
        let attribute = native(
            Stage::Create,
            file.new_attr::<i64>()
                .shape([added.len()])
                .create("inserted"),
        )?;
        native(Stage::Write, attribute.write(&added))?;
        drop(attribute);
        native(Stage::Flush, file.flush())?;
        native(Stage::Close, file.close())?;
    }
    let file = native(Stage::Open, hdf5::File::open(path))?;
    let response = match operation {
        Operation::Count => Response::Count(native(Stage::Names, file.attr_names())?.len()),
        Operation::Lookup(names) => {
            let mut attributes = Vec::new();
            for name in names {
                let attribute = match file.attr(&name) {
                    Ok(attribute) => attribute,
                    Err(error) => {
                        absence::assert_libhdf5_absent(&error, &name);
                        attributes.push(Lookup::Missing(name));
                        continue;
                    }
                };
                let values = native(Stage::AttributeRead, attribute.read_raw::<i64>())?;
                attributes.push(Lookup::Found {
                    name,
                    sum: sum(&values),
                });
            }
            Response::Lookup(attributes)
        }
        Operation::Details | Operation::InsertHuge | Operation::InsertManaged => {
            let mut attributes = Vec::new();
            for name in native(Stage::Names, file.attr_names())? {
                let attribute = native(Stage::AttributeOpen, file.attr(&name))?;
                let dtype = native(Stage::AttributeType, attribute.dtype())?;
                let size = dtype
                    .size()
                    .checked_mul(attribute.size())
                    .ok_or_else(|| Failure::Infrastructure("attribute size overflows".into()))?;
                attributes.push(if dtype.is::<i64>() {
                    Attribute::Integer {
                        name,
                        size,
                        sum: sum(&native(Stage::AttributeRead, attribute.read_raw::<i64>())?),
                    }
                } else {
                    Attribute::Other { name, size }
                });
            }
            Response::Details(attributes)
        }
    };
    native(Stage::Close, file.close())?;
    Ok(response)
}

fn sum(values: &[i64]) -> i64 {
    values.iter().copied().fold(0, i64::wrapping_add)
}

enum Failure {
    Infrastructure(Box<dyn StdError>),
    Native(Rejection),
}

fn native<T>(stage: Stage, result: hdf5::Result<T>) -> Result<T, Failure> {
    result.map_err(|error| match error {
        hdf5::Error::HDF5(stack) => match stack.expand() {
            Ok(stack) => Failure::Native(Rejection {
                stage,
                frames: stack
                    .iter()
                    .map(|frame| Frame {
                        function: frame.func().to_string(),
                        major: format!("{:?}", frame.major_code()),
                        minor: format!("{:?}", frame.minor_code()),
                        description: frame.desc().to_string(),
                    })
                    .collect(),
            }),
            Err(error) => Failure::Infrastructure(Box::new(error)),
        },
        error @ hdf5::Error::Internal(_) => Failure::Infrastructure(Box::new(error)),
    })
}
