# Groups & Attributes

HDF5 files are hierarchical: datasets live inside groups, groups nest inside other groups, and any object (the root, a group, or a dataset) can carry typed metadata in the form of attributes. This page covers building a nested hierarchy with the write API, attaching attributes of several types, and walking the structure back when reading.

!!! tip "Runnable example"
    A complete, runnable version of everything on this page lives in
    [`examples/groups_and_attributes.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/groups_and_attributes.rs).
    Run it with:

    ```bash
    cargo run --example groups_and_attributes
    ```

## Building a hierarchy

Groups are created with a builder API. `FileBuilder::create_group(name)` returns a `GroupBuilder` for a top-level group, and `GroupBuilder::create_group(name)` returns a nested `GroupBuilder` for a sub-group. Each group builder can hold datasets (via `create_dataset`), attributes (via `set_attr`), and further sub-groups.

A group builder is not part of the file until it is finished and attached: call `finish()` to turn it into a `FinishedGroup`, then pass that to its parent's `add_group()`. For a sub-group, the parent is another `GroupBuilder`; for a top-level group, the parent is the `FileBuilder`.

```rust
use hdf5_pure::{AttrValue, File, FileBuilder};

let mut builder = FileBuilder::new();

// A group with its own datasets and attributes.
let mut sensors = builder.create_group("sensors");
sensors.set_attr("location", AttrValue::AsciiString("lab_a".into()));
sensors.set_attr("channels", AttrValue::I64Array(vec![0, 1, 2]));
sensors
    .create_dataset("pressure")
    .with_f32_data(&[101.3, 101.5, 101.4]);
sensors
    .create_dataset("humidity")
    .with_f32_data(&[40.0, 41.5]);

// A nested sub-group. Build it, finish it, and attach it to its parent.
let mut imu = sensors.create_group("imu");
imu.set_attr("model", AttrValue::String("MPU-9250".into()));
imu.create_dataset("accel").with_f64_data(&[0.0, 0.0, 9.81]);
sensors.add_group(imu.finish());

// Attach the top-level group to the file.
builder.add_group(sensors.finish());
```

The pattern is uniform at every level: build, finish, attach. A child must be finished and added before its parent is finished, since `finish()` consumes the builder.

See the [writing guide](writing.md) for the full dataset builder API used inside groups.

!!! note "Group format is fixed, not configurable"
    There is no group creation property list: every group is written with the same fixed, timestamp-free, new-style layout regardless of settings or child count. See [Limitations](../reference/limitations.md#group-creation-property-list-gcpl) for details.

## Attributes

Attributes are small named pieces of metadata. Set them on the root via `FileBuilder::set_attr`, on a group via `GroupBuilder::set_attr`, or on a dataset via `DatasetBuilder::set_attr` (the dataset form is chainable and returns `&mut Self`). The value is an `AttrValue`, an enum covering the common scalar, array, and string encodings:

```rust
// Root-level metadata.
builder.set_attr("title", AttrValue::String("experiment 7".into()));
builder.set_attr("run", AttrValue::I64(7));
builder.set_attr("calibration", AttrValue::F64Array(vec![0.1, 0.2, 0.3]));
```

The `AttrValue` variants and their HDF5 encodings are:

| Variant | HDF5 encoding |
|---|---|
| `AttrValue::F32` / `AttrValue::F32Array` | 32-bit float scalar / array |
| `AttrValue::F64` / `AttrValue::F64Array` | 64-bit float scalar / array |
| `AttrValue::I8` / `AttrValue::I8Array` | Signed 8-bit integer scalar / array |
| `AttrValue::I16` / `AttrValue::I16Array` | Signed 16-bit integer scalar / array |
| `AttrValue::I32` / `AttrValue::I32Array` | Signed 32-bit integer scalar / array |
| `AttrValue::I64` / `AttrValue::I64Array` | Signed 64-bit integer scalar / array |
| `AttrValue::U8` / `AttrValue::U8Array` | Unsigned 8-bit integer scalar / array |
| `AttrValue::U16` / `AttrValue::U16Array` | Unsigned 16-bit integer scalar / array |
| `AttrValue::U32` / `AttrValue::U32Array` | Unsigned 32-bit integer scalar / array |
| `AttrValue::U64` / `AttrValue::U64Array` | Unsigned 64-bit integer scalar / array |
| `AttrValue::String` | UTF-8 string (null-padded) |
| `AttrValue::StringArray` | Array of UTF-8 strings |
| `AttrValue::AsciiString` | Fixed-width ASCII string (charset = ASCII) |
| `AttrValue::AsciiStringArray` | Array of fixed-width ASCII strings (null-padded to the longest element) |
| `AttrValue::VarLenString` / `AttrValue::VarLenStringArray` | Variable-length UTF-8 string scalar / array (uses a global heap collection) |
| `AttrValue::VarLenAsciiString` / `AttrValue::VarLenAsciiStringArray` | Variable-length ASCII string scalar / array (uses a global heap collection) |
| `AttrValue::VarLenAsciiCharArray` | MATLAB's array of variable-length ASCII strings: a VLEN *sequence of one-byte strings* (uses a global heap collection) |

!!! note
    `AttrValue::AsciiString`, `AttrValue::AsciiStringArray`, and `AttrValue::VarLenAsciiCharArray` exist for compatibility with MATLAB and matio, which expect fixed-width or variable-length ASCII rather than UTF-8 for certain conventional attributes. See the [data types reference](../reference/data-types.md) for the full type mapping.

    The two variable-length families differ in datatype, not in bytes. `VarLenString` and its siblings write `H5T_STRING` with `STRSIZE = H5T_VARIABLE` — what h5py and the C library write, and what h5py reads back as a `str` and the C library as a `char *`. `VarLenAsciiCharArray` writes `H5T_VLEN { H5T_STRING { STRSIZE = 1 } }`, which MATLAB and matio expect for `MATLAB_fields` and its neighbours ([#383](https://github.com/CramBL/hdf5-pure/issues/383)).

## Reading the hierarchy back

Open the file and start from `File::root()`, which returns a `Group` for the root. From any `Group` you can list its contents and read its attributes:

- `groups()` returns the names of child groups (`Vec<String>`).
- `datasets()` returns the names of child datasets (`Vec<String>`).
- `iter_groups()` and `iter_datasets()` return those same children as opened handles paired with their names, walking the group once where opening each name separately re-walks it per member. Use them when the members themselves are what you want; use the name lists when a listing is.
- `attrs()` returns the attributes as a `HashMap<String, AttrValue>`.
- `attr_datatypes()` returns their on-disk datatypes as a `HashMap<String, Datatype>`.

```rust
let file = File::from_bytes(builder.finish().unwrap()).unwrap();

let root = file.root();
let root_attrs = root.attrs().unwrap(); // HashMap<String, AttrValue>

let sensors = file.group("sensors").unwrap();
println!("child groups: {:?}", sensors.groups().unwrap());   // ["imu"]
println!("datasets:     {:?}", sensors.datasets().unwrap()); // ["humidity", "pressure"]
println!("attributes:   {:?}", sensors.attrs().unwrap());
```

`File::group(path)` resolves a group by path, and `Group::group(name)` resolves a child relative to that group. Both fail with `Error::NotAGroup` when the name reaches something that is not a group — a dataset, or a committed datatype — the way `Error::NotADataset` reports the mirror case for `dataset()`; a name that reaches nothing at all is a `FormatError::PathNotFound` instead. The same error reports a component *along* a path, since resolving `a/b/c` opens `a` and then `a/b` to look inside them, and it names that component's own path rather than the one asked for: a dataset at `a/b` gives `Error::NotAGroup("a/b")` from `File::group("a/b/c")` and from `File::dataset("a/b/c")` alike. The names returned by `groups()` and `datasets()` are not sorted in any guaranteed order, so sort them yourself if you need a stable listing.

### Reading an attribute value

An attribute reads back as the variant it was written from: the dataspace kind distinguishes a scalar from a one-element array, the datatype's charset selects the `Ascii` variants, and a number's width selects among `I8` … `U64` and `F32` / `F64`. A `VarLenAsciiCharArray` of one element stays a `VarLenAsciiCharArray`, an `AsciiString` does not arrive as a `String`, and a 16-bit integer does not arrive as an `I64`. A variable-length string is not a fixed-width one either: an h5py or C-library attribute arrives as `VarLenString` or `VarLenAsciiString` (or their array forms), so setting the value back writes the datatype it was found in rather than a fixed-width slot.

A fixed-width string carries its width the same way. `AttrValue::ascii_string_sized("ok", 64)` declares a 64-byte slot — `H5T_C_S1` plus `H5Tset_size(64)` — and a slot whose width is not the one its content implies reads back as `AsciiStringSized` (or `StringSized`, or the array forms) carrying that width, so writing the value back reproduces the slot rather than shrinking it to the content. A slot sized exactly to its content reads back as the plain `AsciiString`, which writes the same bytes.

That fidelity means several variants can carry the same logical value, so match on the variant only when the encoding is what you care about. Otherwise use the accessors, each of which spans every variant that can hold the shape it names:

```rust
let attrs = file.root().attrs().unwrap();

// Any single string, either charset, scalar or one-element array.
let class: Option<&str> = attrs.get("MATLAB_class").and_then(AttrValue::as_str);

// Every element, with a scalar reading as one element.
let fields: Option<Vec<&str>> = attrs.get("MATLAB_fields").and_then(AttrValue::as_strings);
```

`as_i64`, `as_u64`, `as_f64`, `to_i64s`, `to_u64s` and `to_f64s` do the same for numbers. The prefix states the cost: `as_*` borrows or copies, `to_*` allocates. `as_i64` widens the narrower integer variants and reports `None` for a value above `i64::MAX` rather than wrapping it — per element, so the same holds for `to_i64s` at any length. The float accessors do not convert integers, so a caller that accepts either asks for both.

What a read cannot recover, because `AttrValue` has no way to express it. Each of these reads correctly, and each would come back differently if it were *rewritten from the value* — which is why [`repack`](repack.md) copies an attribute's encoding rather than rebuilding it from one:

- **Byte order and precision.** Every numeric variant writes back little-endian at its full width, so a big-endian attribute, one storing fewer bits than its bytes hold, or a float laid out other than the way IEEE 754 lays it out, reads correctly but would be re-encoded in this crate's own layout.
- **Unrepresentable integer widths.** A number keeps its width at 1, 2, 4 and 8 bytes; an integer width with no Rust integer of its own — 3 bytes, say — widens to 64-bit.
- **A scalar in MATLAB's sequence-of-one-byte-strings shape.** Its array form reads as `VarLenAsciiCharArray`, the form MATLAB writes; a scalar in that shape has no variant and reads as `AsciiString`, which writes a fixed-width slot. A standard variable-length string keeps its datatype at either arity.
- **Rank.** Every array variant is one-dimensional, so a rank-2 attribute reads as its elements flattened.
- **String padding.** A fixed-width string reports its content and the `STRSIZE` it was stored at, but not which padding rule filled the bytes past the content: every variant writes back `NULLPAD`.
- **Enumeration member names.** An enum attribute decodes through its integer base, so its codes arrive and its labels do not.

The variant may become **more specific** in a future release as `AttrValue` grows further variants — the numeric widths landed that way — so match with a `_` arm, which the `#[non_exhaustive]` enum requires anyway, or read through the accessors, which are unaffected by such a change.

One case is a refusal rather than a loss of fidelity: a number **wider than 8 bytes** has no `AttrValue` at all. The typed numeric readers model an element as a 64-bit word taken from its leading eight bytes, so a 9-byte integer holding 2^64 would read back as `0` — indistinguishable from one that holds zero — and big-endian it would read back as 2^56, since those leading bytes are the *most* significant ones. `attrs()` omits such an attribute instead; `attr_datatypes()` still reports its full width, which is how it can be told from an attribute that is not there.

### Reading an attribute's datatype

`attr_datatypes()` reports the on-disk [`Datatype`](../reference/data-types.md#the-datatype-model) of every attribute, keyed by name. It is the type channel to `attrs()`'s value channel, the pair a dataset already has in `datatype()` and its `read_*` methods, and it is where the *datatype* entries in the list above — byte order and precision, string padding and declared width, enumeration member names — can still be read.

**Rank is not among them.** An attribute's rank lives in its dataspace, which nothing public exposes, so a rank-2 attribute reads as a flat array from `attrs()` with no way to recover its shape from either channel.

It reports **every** attribute message, including the ones `attrs()` omits because no `AttrValue` can carry them, so a name missing from that map can be told from one the object does not have.

A **committed** (shared) type, created with `H5Tcommit`, is stored on the attribute as a reference to the type's own object header rather than as the type itself, and both this and `Dataset::datatype()` follow that reference and report the type it names. netCDF-4 user-defined types and h5py's `f["t"] = np.dtype(...)` reach a file this way. What neither channel reports is the type's *name*: two attributes sharing `/mytype` give the same `Datatype` as one that spells it out inline.

A boolean attribute needs both channels. The C library gives `H5T_NATIVE_HBOOL` — what h5py writes for every `np.bool_` — an `enum[FALSE, TRUE]` over an 8-bit base, so the value arrives as `0` or `1`, indistinguishable from an `i8`, and only the datatype records which it was:

```rust
use hdf5_pure::Datatype;

let root = file.root();
let is_bool = matches!(
    root.attr_datatypes().unwrap().get("success"),
    Some(Datatype::Enumeration { base_type, members, .. })
        if matches!(**base_type, Datatype::FixedPoint { size: 1, .. })
            && members.iter().any(|m| m.name == "TRUE")
);
let value = root.attrs().unwrap().get("success").and_then(AttrValue::as_i64);
```

### Addressing datasets

Datasets are addressable two ways: by full path from the file, or by name from their parent group. Both resolve to the same dataset.

```rust
// Full path from the file.
let accel = file.dataset("sensors/imu/accel").unwrap();
println!("{:?}", accel.read_f64().unwrap()); // [0.0, 0.0, 9.81]

// By name, relative to the parent group.
let imu = file.group("sensors/imu").unwrap();
let accel = imu.dataset("accel").unwrap();
```

See the [reading guide](reading.md) for the dataset read API (`read_f64`, `shape`, and friends).

## MATLAB struct convention

The ASCII attribute variants exist primarily so that groups can follow MATLAB's struct convention: a struct is a group carrying `MATLAB_class = "struct"` and a `MATLAB_fields` attribute (typically `AttrValue::VarLenAsciiCharArray`) listing the field names, with each field stored as a child dataset tagged with its own `MATLAB_class`. See the [MATLAB interop page](../interop/matlab.md) for the full convention and worked examples.
