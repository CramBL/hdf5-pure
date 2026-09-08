---
description: Pure-Rust HDF5 reader, writer, and in-place editor. No C dependencies, no build scripts, WASM-compatible.
hide:
  - navigation
  - toc
---

<div class="h5-hero" markdown>

# hdf5-pure

<p class="h5-hero__tag">Pure-Rust HDF5 — read, write, and edit files in place.</p>

<p>No C dependencies. No build scripts. WASM-compatible. Interoperable with the
reference HDF5 C library, h5py, and MATLAB.</p>

[Get started](getting-started/installation.md){ .md-button .md-button--primary }
[Quick start](getting-started/quickstart.md){ .md-button }
[View on GitHub](https://github.com/CramBL/hdf5-pure){ .md-button }

<div class="h5-pills">
  <span>read</span>
  <span>write</span>
  <span>edit in place</span>
  <span>streaming</span>
  <span>SWMR</span>
  <span>compression</span>
  <span>compound types</span>
  <span>ndarray</span>
  <span>MATLAB v7.3</span>
  <span>no_std</span>
  <span>WASM</span>
</div>

</div>

<p align="center">
<a href="https://crates.io/crates/hdf5-pure"><img alt="crates.io" src="https://img.shields.io/crates/v/hdf5-pure.svg?logo=rust&color=0e7490"></a>
<a href="https://docs.rs/hdf5-pure"><img alt="docs.rs" src="https://img.shields.io/docsrs/hdf5-pure?logo=docsdotrs&color=0e7490"></a>
<a href="https://github.com/CramBL/hdf5-pure/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/CramBL/hdf5-pure/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://github.com/CramBL/hdf5-pure/blob/main/README.md#license"><img alt="License: MIT or Apache-2.0" src="https://img.shields.io/crates/l/hdf5-pure.svg?color=0e7490"></a>
</p>

`hdf5-pure` is a zero-C-dependency crate for creating, reading, and editing
[HDF5](https://www.hdfgroup.org/solutions/hdf5/) files. It builds on stable Rust
with no C toolchain, runs in the browser via WebAssembly, and writes files that
the reference HDF5 C library, h5py, and MATLAB read without conversion.

## What it does

<div class="grid cards" markdown>

-   :material-pencil-box-outline:{ .lg .middle } __Write & read__

    ---

    Build files with datasets, groups, attributes, and nested hierarchies — in
    memory or on disk. Read v0–v3 superblocks, v1/v2 object headers, and
    contiguous, chunked, or compact storage.

    [:octicons-arrow-right-24: Writing files](guide/writing.md)

-   :material-file-edit-outline:{ .lg .middle } __Edit in place__

    ---

    Add, delete, and copy objects in an existing file without rewriting it. The
    cost is proportional to what changes, and a failed commit leaves the file
    valid.

    [:octicons-arrow-right-24: Editing in place](guide/editing.md)

-   :material-zip-box-outline:{ .lg .middle } __Compression & filters__

    ---

    Deflate, shuffle, scale-offset (lossless integer / lossy float), and an
    optional pure-Rust ZFP codec — all built-in HDF5 filters, so the files stay
    portable.

    [:octicons-arrow-right-24: Compression & filters](guide/compression.md)

-   :material-fast-forward-outline:{ .lg .middle } __Streaming & SWMR__

    ---

    Read files too large to buffer with on-demand chunk fetching, and append to
    unlimited datasets while other processes read them concurrently.

    [:octicons-arrow-right-24: Streaming](guide/streaming.md) ·
    [SWMR](guide/swmr.md)

-   :material-matrix:{ .lg .middle } __MATLAB v7.3__

    ---

    Read and write `.mat` v7.3 files: userblocks, the MATLAB struct convention,
    and a serde path that maps Rust structs straight to MATLAB variables.

    [:octicons-arrow-right-24: MATLAB v7.3](interop/matlab.md)

-   :material-language-rust:{ .lg .middle } __WASM & no_std__

    ---

    Pure Rust, no C toolchain. The in-memory API runs in the browser via
    WebAssembly; turn default features off to compile for bare-metal `no_std`.

    [:octicons-arrow-right-24: Portability](interop/portability.md)

</div>

## A first taste

=== "Write"

    ```rust
    use hdf5_pure::{AttrValue, FileBuilder};

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("temperature")
        .with_f64_data(&[22.5, 23.1, 21.8])
        .set_attr("unit", AttrValue::AsciiString("degC".into()));
    builder.set_attr("version", AttrValue::I64(2));

    // In memory (WASM-friendly) — or `builder.write("out.h5")` for a file.
    let bytes: Vec<u8> = builder.finish().unwrap();
    ```

=== "Read"

    ```rust
    use hdf5_pure::File;

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("temperature").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![3]);
    assert_eq!(ds.read_f64().unwrap(), vec![22.5, 23.1, 21.8]);

    let version = file.root().attrs().unwrap().get("version").cloned();
    assert_eq!(version, Some(hdf5_pure::AttrValue::I64(2)));
    ```

Ready to dig in? Start with [Installation](getting-started/installation.md) and
the [Quick Start](getting-started/quickstart.md), then browse the
[Guide](guide/writing.md). Every page mirrors a runnable example under
[`examples/`](https://github.com/CramBL/hdf5-pure/tree/main/examples).
