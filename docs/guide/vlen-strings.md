HDF5 strings come in two on-disk shapes: fixed-length (a fixed byte width per element, padded) and variable-length (each element stores a pointer into the file's global heap). This page covers reading both, with a focus on bounding and streaming variable-length reads so a single dataset cannot exhaust memory before you know how big it is.

For the broader reading API see [Reading files](crate::_guide::reading). For the streaming backend used in several examples here, see [Streaming large files](crate::_guide::streaming). Writing variable-length strings as *attributes* is a separate facility (the [`AttrValue::VarLenString`](crate::AttrValue::VarLenString) family), covered in [Groups and attributes](crate::_guide::groups_attributes).

## Reading any string dataset

[`Dataset::read_string`](crate::Dataset::read_string) reads both fixed-length and variable-length string datasets and returns a `Vec<String>`. You do not need to know which encoding the file uses: the method inspects the datatype and dispatches accordingly.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("strings.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("names").with_vlen_strings(&["ada", "grace", "katherine"]);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open(&path)?;
let names = file.dataset("names")?.read_string()?;

for name in &names {
    println!("{name}");
}
# assert_eq!(names, vec!["ada", "grace", "katherine"]);
# Ok::<(), hdf5_pure::Error>(())
```

This is the right call for a dataset of a known, modest size. For large variable-length datasets, the methods below let you measure and bound the read first.

## Measuring payload size before allocation

A variable-length string dataset's in-memory size is not implied by its shape: each element points at a separately sized run of bytes in the global heap. [`Dataset::vlen_string_payload_size`](crate::Dataset::vlen_string_payload_size) reports the total payload size, in bytes, that a full read would materialize, without building the `Vec<String>`:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("strings.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("names").with_vlen_strings(&["ada", "grace", "katherine"]);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open(&path)?;
let dataset = file.dataset("names")?;

let payload_bytes = dataset.vlen_string_payload_size()?;
println!("{payload_bytes} bytes of string payload");
# assert_eq!(payload_bytes, 17);
# Ok::<(), hdf5_pure::Error>(())
```

The payload figure counts only the bytes referenced by the variable-length elements. It excludes the `Vec<String>` and `String` allocation metadata that Rust adds when the strings are actually materialized. This mirrors the role of HDF5's `H5Dvlen_get_buf_size`.

[`vlen_string_payload_size`](crate::Dataset::vlen_string_payload_size) is defined only for variable-length string datasets. Called on any other datatype it returns an error.

## Bounding a read with `VlenStringReadOptions`

When a dataset may be larger than you are willing to allocate, pass [`VlenStringReadOptions`](crate::VlenStringReadOptions) to bound the read. Both limits are checked *before* any string payload is materialized, so an oversized dataset fails before the first allocation.

| Builder method | Bounds | Notes |
|---|---|---|
| [`VlenStringReadOptions::new()`](crate::VlenStringReadOptions::new) | none | No limits (equivalent to the default). |
| [`.with_max_elements(n)`](crate::VlenStringReadOptions::with_max_elements) | Number of VL elements | Checked against the dataspace element count. |
| [`.with_max_payload_bytes(n)`](crate::VlenStringReadOptions::with_max_payload_bytes) | Total referenced payload bytes | Excludes Rust container metadata. |

The options are a `const`-constructible builder, so a limit can be assembled in a `const` context:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("strings.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("names").with_vlen_strings(&["ada", "grace", "katherine"]);
# builder.write(&path)?;
use hdf5_pure::{File, VlenStringReadOptions};

let file = File::open(&path)?;
let dataset = file.dataset("names")?;

let options = VlenStringReadOptions::new()
    .with_max_elements(100_000)
    .with_max_payload_bytes(64 * 1024 * 1024);

let names = dataset.read_vlen_strings(options)?;
# assert_eq!(names, vec!["ada", "grace", "katherine"]);
# Ok::<(), hdf5_pure::Error>(())
```

[`Dataset::read_vlen_strings`](crate::Dataset::read_vlen_strings) reads the whole dataset into a `Vec<String>` like [`read_string`](crate::Dataset::read_string), but enforces the limits. If either bound is exceeded the call returns an error and nothing is allocated for the payload.

## Visiting strings one at a time

To process a large dataset without ever holding every decoded string at once, use [`Dataset::visit_vlen_strings`](crate::Dataset::visit_vlen_strings). It takes the same [`VlenStringReadOptions`](crate::VlenStringReadOptions) and a closure that is invoked once per element with a `&str` borrowed for the duration of the call:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("strings.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("names").with_vlen_strings(&["ada", "grace", "katherine"]);
# builder.write(&path)?;
use hdf5_pure::{File, VlenStringReadOptions};

let file = File::open_streaming(&path)?;
let dataset = file.dataset("names")?;

let mut longest = 0usize;
dataset.visit_vlen_strings(VlenStringReadOptions::new(), |value| {
    longest = longest.max(value.len());
})?;

println!("longest string: {longest} bytes");
# assert_eq!(longest, 9);
# Ok::<(), hdf5_pure::Error>(())
```

Because the slice handed to the closure is valid only for that call, this is the lowest-memory way to scan a variable-length dataset: aggregate, filter, or stream each string out as it arrives. [`read_vlen_strings`](crate::Dataset::read_vlen_strings) is implemented on top of [`visit_vlen_strings`](crate::Dataset::visit_vlen_strings) by pushing each value onto a `Vec`.

## How payloads are fetched

Variable-length elements reference objects in the file's global heap, and many elements typically share a heap collection. The reader indexes each shared global heap collection once per read and then fetches only the object payloads the dataset's elements actually reference, so duplicated or shared storage is not read repeatedly.

This holds for both backends. Everything on this page works the same with [`File::open`](crate::File::open) (whole file in memory) and [`File::open_streaming`](crate::File::open_streaming) (metadata and chunks fetched on demand), so the bounding and streaming-visit pattern is exactly what you want for a multi-gigabyte file opened with the streaming reader.

## Scope

These methods cover *reading* strings. On the write side, [`DatasetBuilder::with_ascii_strings`](crate::DatasetBuilder::with_ascii_strings) and [`with_strings`](crate::DatasetBuilder::with_strings) write the fixed-width kind (see [Strings](crate::_guide::writing#strings)). The variable-length kind has two facilities: [`DatasetBuilder::with_vlen_strings`](crate::DatasetBuilder::with_vlen_strings) writes a contiguous variable-length UTF-8 string dataset, and the attribute variants [`AttrValue::VarLenString`](crate::AttrValue::VarLenString), [`VarLenStringArray`](crate::AttrValue::VarLenStringArray), [`VarLenAsciiString`](crate::AttrValue::VarLenAsciiString) and [`VarLenAsciiStringArray`](crate::AttrValue::VarLenAsciiStringArray) write a variable-length string as an attribute, with [`AttrValue::VarLenAsciiCharArray`](crate::AttrValue::VarLenAsciiCharArray) writing MATLAB's sequence-of-one-byte-strings shape (see [Groups and attributes](crate::_guide::groups_attributes)). Both of those store their payloads in the file's global heap, split across as many collections as their element count needs. Chunked, filtered, and resizable variable-length-string datasets write too: their heap collections are placed ahead of the chunk data so the element references carry real addresses before a chunk is compressed. An existing dataset's strings can also be replaced in place through [`Dataset::write_staged`](crate::Dataset::write_staged) with the same builder method, which places a fresh collection for the new strings, and [Editing files](crate::_guide::editing) covers that path. For how string datatypes map between HDF5 and this crate, see [Element types](crate::DatasetBuilder#element-types).
