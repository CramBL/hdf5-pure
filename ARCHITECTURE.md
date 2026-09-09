# Architecture

TODO: describe the layers once the `no_std` core and the shell are separated.

## The pipeline

Internally the crate is layered. Lower layers know nothing about the layers
above them. Data flows up from raw bytes to typed values.

| Layer | Responsibility |
|---|---|
| **Primitives** | Errors, checked integer conversions, checksums, byte sources, the file signature |
| **Format structures** | Superblocks, object headers, datatypes, dataspaces, data layouts, links, heaps (local, global, fractal), B-trees (v1/v2), symbol tables |
| **Filters** | The filter pipeline plus deflate, shuffle, scale-offset, and the optional ZFP codec |
| **Engine** | The chunked read/write core, chunk indexes (B-tree v1, fixed array, extensible array), the chunk cache, the file writer, attribute and group machinery |
| **High-level API** | `File`, `Dataset`, `Group`, `FileBuilder`, `StagedGroup`, and the `ndarray` integration |
| **MATLAB v7.3** | The serde-based `.mat` reader/writer, built on top of the engine and the high-level API |

The read and write paths are mutually recursive - index structures call back
into the writer, the writer drives chunked I/O, and so on - so the engine is a
single strongly-connected component that cannot be cleanly split into separate
"reader" and "writer" crates without dependency inversion.
