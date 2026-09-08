# Quick Start

This is the shortest useful path through `hdf5-pure`: build a file **in memory**,
then read it back. No filesystem and no C library are involved, which is exactly
what makes the same code run in a browser via WASM.

It mirrors the runnable
[`quickstart` example](https://github.com/CramBL/hdf5-pure/blob/main/examples/quickstart.rs).
You can run it directly from a clone:

```bash
cargo run --example quickstart
```

## Write

A [`FileBuilder`](../guide/writing.md) accumulates datasets, groups, and
attributes, then serializes them to an HDF5 file image.

```rust
use hdf5_pure::{AttrValue, FileBuilder};

let mut builder = FileBuilder::new();

// A dataset. The shape defaults to `[len]`, so `with_shape` is optional for a
// flat 1-D array; it is shown here for clarity.
builder
    .create_dataset("temperature")
    .with_f64_data(&[22.5, 23.1, 21.8])
    .with_shape(&[3])
    .set_attr("unit", AttrValue::AsciiString("degC".into()));

// An attribute on the root group.
builder.set_attr("version", AttrValue::I64(2));

// `finish()` returns the file image as bytes; `write(path)` would put them on
// disk instead. The in-memory form is what makes this WASM-friendly.
let bytes: Vec<u8> = builder.finish().expect("serialize file");
```

!!! tip "In memory vs. on disk"

    `builder.finish()` returns a `Vec<u8>` you can hand to a network call,
    embed, or hash. `builder.write("output.h5")` does the same serialization but
    streams it to a path. The two share all the same builder code.

## Read

[`File`](../guide/reading.md) parses a file image (from bytes or a path) and
gives you typed access to datasets and attributes.

```rust
use hdf5_pure::File;

let file = File::from_bytes(bytes).expect("parse file");

let ds = file.dataset("temperature").expect("open dataset");
println!("shape: {:?}", ds.shape().unwrap());      // [3]
println!("data:  {:?}", ds.read_f64().unwrap());   // [22.5, 23.1, 21.8]
println!("unit:  {:?}", ds.attrs().unwrap().get("unit"));

let root_attrs = file.root().attrs().unwrap();
println!("version: {:?}", root_attrs.get("version")); // Some(I64(2))
```

`File::from_bytes` reads a complete in-memory image. To read a file from disk,
use `File::open("output.h5")`; to read one too large to buffer, use
[`File::open_streaming`](../guide/streaming.md). The reading API is identical
across all three.

## Where to go next

<div class="grid cards" markdown>

-   __Build richer files__

    ---

    Datasets of every scalar type, nested groups, and typed attributes.

    [:octicons-arrow-right-24: Writing files](../guide/writing.md) ·
    [Groups & attributes](../guide/groups-attributes.md)

-   __Read what others wrote__

    ---

    Open files from the C library, h5py, or MATLAB and walk their contents.

    [:octicons-arrow-right-24: Reading files](../guide/reading.md)

-   __Shrink storage__

    ---

    Chunking plus deflate, shuffle, scale-offset, or ZFP.

    [:octicons-arrow-right-24: Compression & filters](../guide/compression.md)

-   __Change a file without rewriting it__

    ---

    Add, copy, and delete objects in place; reclaim space with `repack`.

    [:octicons-arrow-right-24: Editing in place](../guide/editing.md)

</div>
