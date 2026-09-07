# Installation

`hdf5-pure` is a regular Cargo dependency with no system libraries, no C
toolchain, and no build step to configure.

## Add the crate

```bash
cargo add hdf5-pure
```

Or add it to `Cargo.toml` by hand:

```toml
[dependencies]
hdf5-pure = "0.14"
```

That pulls the default feature set — `std`, `checksum`, and `deflate` — which
covers file I/O, the high-level reader/writer API, and deflate compression.

!!! note "Toolchain"

    The crate uses Rust **edition 2024**, so it needs a 2025-era toolchain
    (Rust 1.85 or newer). It builds on stable; no nightly features are required.

## Choosing features

Most functionality beyond the core read/write API is gated behind a Cargo
feature so you only compile what you use:

```toml
[dependencies]
hdf5-pure = { version = "0.14", features = ["ndarray", "serde", "zfp"] }
```

| Feature | Default | Enables |
|---|:---:|---|
| `std` | ✅ | File I/O and the high-level reader API |
| `checksum` | ✅ | Jenkins hash validating checksummed metadata |
| `deflate` | ✅ | Deflate (zlib) compression, pure-Rust backend |
| `serde` | | Read/write MATLAB v7.3 `.mat` files via serde |
| `ndarray` | | N-dimensional array I/O via the `ndarray` crate |
| `zfp` | | Pure-Rust ZFP fixed-rate compression (HDF5 filter 32013) |
| `fast-deflate` | | zlib-ng deflate backend (faster, links C) |
| `provenance` | | SHA-256 data provenance tracking |

See the [Cargo Features reference](../reference/features.md) for the full table
and the trade-offs of each.

## WebAssembly and `no_std`

The crate is pure Rust, so it builds for the browser with no extra toolchain.
`std` is available on `wasm32-unknown-unknown`, so keep the default features
(which include `std`) and just add the target:

```bash
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown
```

In a WASM build you use the **in-memory** API: `FileBuilder::finish` returns the
file as a `Vec<u8>` and `File::from_bytes` parses one, neither of which touches
a filesystem. The path-based entry points (`File::open`, `FileBuilder::write`,
`File::open_rw`, `File::open_swmr_writer`) compile but cannot reach a filesystem at runtime in
the browser.

!!! note "Bare-metal `no_std`"

    With `default-features = false` the crate is `#![no_std]` and depends only
    on `alloc`, and it compiles for freestanding targets (CI builds
    `thumbv7em-none-eabi`):

    ```toml
    [dependencies]
    hdf5-pure = { version = "0.14", default-features = false, features = ["checksum"] }
    ```

    The high-level `File` / `FileBuilder` API is `std`-gated, so a
    pure-`no_std` build exposes the lower-level datatype and builder primitives
    rather than the whole-file reader and writer. See
    [Portability](../interop/portability.md) for the full breakdown of what each
    target supports.

## Verify the install

Drop this into a binary crate and run it — it builds a file in memory and reads
it back, touching no filesystem:

```rust
use hdf5_pure::{File, FileBuilder};

fn main() {
    let mut builder = FileBuilder::new();
    builder.create_dataset("x").with_f64_data(&[1.0, 2.0, 3.0]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let values = file.dataset("x").unwrap().read_f64().unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
    println!("hdf5-pure is working: {values:?}");
}
```

Next, walk through the [Quick Start](quickstart.md).

## Building this documentation

The site you are reading is built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).
To preview it locally, with [uv](https://docs.astral.sh/uv/) and [just](https://just.systems) installed:

```bash
just docs::serve   # http://127.0.0.1:8000
```
