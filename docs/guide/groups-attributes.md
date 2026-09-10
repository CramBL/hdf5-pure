HDF5 files are hierarchical: datasets live inside groups, groups nest inside other groups, and any object (the root, a group, or a dataset) can carry typed metadata in the form of attributes. This page covers building a nested hierarchy with the write API, attaching attributes of several types, and walking the structure back when reading.

A complete, runnable version of everything on this page lives in [`examples/groups_and_attributes.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/groups_and_attributes.rs). Run it with:

```console
$ cargo run --example groups_and_attributes
```

## Building a hierarchy

Groups are created with a builder API. [`FileBuilder::create_group(name)`](crate::FileBuilder::create_group) returns a [`GroupBuilder`](crate::GroupBuilder) for a top-level group, and [`GroupBuilder::create_group(name)`](crate::GroupBuilder::create_group) returns a nested [`GroupBuilder`](crate::GroupBuilder) for a sub-group. Each group builder can hold datasets (via [`create_dataset`](crate::GroupBuilder::create_dataset)), attributes (via [`set_attr`](crate::GroupBuilder::set_attr)), and further sub-groups.

A group builder is not part of the file until it is finished and attached: call [`finish()`](crate::GroupBuilder::finish) to turn it into a [`FinishedGroup`](crate::FinishedGroup), then pass that to its parent's [`add_group()`](crate::GroupBuilder::add_group). The parent is another [`GroupBuilder`](crate::GroupBuilder) for a sub-group, and the [`FileBuilder`](crate::FileBuilder) for a top-level group.

```rust
use hdf5_pure::{AttrValue, FileBuilder};

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
# let file = hdf5_pure::File::from_bytes(builder.finish()?)?;
# assert_eq!(file.group("sensors")?.groups()?, vec!["imu"]);
# Ok::<(), hdf5_pure::Error>(())
```

The pattern is uniform at every level: build, finish, attach. A child must be finished and added before its parent is finished, since [`finish()`](crate::GroupBuilder::finish) consumes the builder.

See [Writing files](crate::_guide::writing) for the full dataset builder API used inside groups.

There is no group creation property list: every group is written with the same fixed, timestamp-free, new-style layout regardless of settings or child count. The limitations reference has the details.

## Attributes

Attributes are small named pieces of metadata. Set them on the root via [`FileBuilder::set_attr`](crate::FileBuilder::set_attr), on a group via [`GroupBuilder::set_attr`](crate::GroupBuilder::set_attr), or on a dataset via [`DatasetBuilder::set_attr`](crate::DatasetBuilder::set_attr) (the dataset form is chainable and returns `&mut Self`). The value is an [`AttrValue`](crate::AttrValue), an enum covering the common scalar, array, and string encodings:

```rust
# use hdf5_pure::{AttrValue, FileBuilder};
# let mut builder = FileBuilder::new();
// Root-level metadata.
builder.set_attr("title", AttrValue::String("experiment 7".into()));
builder.set_attr("run", AttrValue::I64(7));
builder.set_attr("calibration", AttrValue::F64Array(vec![0.1, 0.2, 0.3]));
# let file = hdf5_pure::File::from_bytes(builder.finish()?)?;
# assert_eq!(file.root().attrs()?["run"], AttrValue::I64(7));
# Ok::<(), hdf5_pure::Error>(())
```

The [`AttrValue`](crate::AttrValue) variants and their HDF5 encodings are:

| Variant | HDF5 encoding |
|---|---|
| [`AttrValue::F32`](crate::AttrValue::F32) / [`AttrValue::F32Array`](crate::AttrValue::F32Array) | 32-bit float scalar / array |
| [`AttrValue::F64`](crate::AttrValue::F64) / [`AttrValue::F64Array`](crate::AttrValue::F64Array) | 64-bit float scalar / array |
| [`AttrValue::I8`](crate::AttrValue::I8) / [`AttrValue::I8Array`](crate::AttrValue::I8Array) | Signed 8-bit integer scalar / array |
| [`AttrValue::I16`](crate::AttrValue::I16) / [`AttrValue::I16Array`](crate::AttrValue::I16Array) | Signed 16-bit integer scalar / array |
| [`AttrValue::I32`](crate::AttrValue::I32) / [`AttrValue::I32Array`](crate::AttrValue::I32Array) | Signed 32-bit integer scalar / array |
| [`AttrValue::I64`](crate::AttrValue::I64) / [`AttrValue::I64Array`](crate::AttrValue::I64Array) | Signed 64-bit integer scalar / array |
| [`AttrValue::U8`](crate::AttrValue::U8) / [`AttrValue::U8Array`](crate::AttrValue::U8Array) | Unsigned 8-bit integer scalar / array |
| [`AttrValue::U16`](crate::AttrValue::U16) / [`AttrValue::U16Array`](crate::AttrValue::U16Array) | Unsigned 16-bit integer scalar / array |
| [`AttrValue::U32`](crate::AttrValue::U32) / [`AttrValue::U32Array`](crate::AttrValue::U32Array) | Unsigned 32-bit integer scalar / array |
| [`AttrValue::U64`](crate::AttrValue::U64) / [`AttrValue::U64Array`](crate::AttrValue::U64Array) | Unsigned 64-bit integer scalar / array |
| [`AttrValue::String`](crate::AttrValue::String) | UTF-8 string (null-padded) |
| [`AttrValue::StringArray`](crate::AttrValue::StringArray) | Array of UTF-8 strings |
| [`AttrValue::AsciiString`](crate::AttrValue::AsciiString) | Fixed-width ASCII string (charset = ASCII) |
| [`AttrValue::AsciiStringArray`](crate::AttrValue::AsciiStringArray) | Array of fixed-width ASCII strings (null-padded to the longest element) |
| [`AttrValue::VarLenString`](crate::AttrValue::VarLenString) / [`AttrValue::VarLenStringArray`](crate::AttrValue::VarLenStringArray) | Variable-length UTF-8 string scalar / array (uses a global heap collection) |
| [`AttrValue::VarLenAsciiString`](crate::AttrValue::VarLenAsciiString) / [`AttrValue::VarLenAsciiStringArray`](crate::AttrValue::VarLenAsciiStringArray) | Variable-length ASCII string scalar / array (uses a global heap collection) |
| [`AttrValue::VarLenAsciiCharArray`](crate::AttrValue::VarLenAsciiCharArray) | MATLAB's array of variable-length ASCII strings: a VLEN *sequence of one-byte strings* (uses a global heap collection) |

[`AttrValue::AsciiString`](crate::AttrValue::AsciiString), [`AttrValue::AsciiStringArray`](crate::AttrValue::AsciiStringArray), and [`AttrValue::VarLenAsciiCharArray`](crate::AttrValue::VarLenAsciiCharArray) exist for compatibility with MATLAB and matio, which expect fixed-width or variable-length ASCII for certain conventional attributes. The data types reference has the full type mapping.

The two variable-length families differ in datatype, not in bytes. [`VarLenString`](crate::AttrValue::VarLenString) and its siblings write `H5T_STRING` with `STRSIZE = H5T_VARIABLE`, what h5py and the C library write, and what h5py reads back as a `str` and the C library as a `char *`. [`VarLenAsciiCharArray`](crate::AttrValue::VarLenAsciiCharArray) writes `H5T_VLEN { H5T_STRING { STRSIZE = 1 } }`, which MATLAB and matio expect for `MATLAB_fields` and its neighbors ([#383](https://github.com/CramBL/hdf5-pure/issues/383)).

## Reading the hierarchy back

Open the file and start from [`File::root()`](crate::File::root), which returns a [`Group`](crate::Group) for the root. From any [`Group`](crate::Group) you can list its contents and read its attributes:

- [`groups()`](crate::Group::groups) returns the names of child groups (`Vec<String>`).
- [`datasets()`](crate::Group::datasets) returns the names of child datasets (`Vec<String>`).
- [`iter_groups()`](crate::Group::iter_groups) and [`iter_datasets()`](crate::Group::iter_datasets) return those same children as opened handles paired with their names, walking the group once where opening each name separately re-walks it per member. Use them when the members themselves are what you want, and the name lists when a listing is.
- [`attrs()`](crate::Group::attrs) returns the attributes as a `HashMap<String, AttrValue>`.
- [`attr_datatypes()`](crate::Group::attr_datatypes) returns their on-disk datatypes as a `HashMap<String, Datatype>`.

```rust
# use hdf5_pure::{AttrValue, File, FileBuilder};
# let mut builder = FileBuilder::new();
# let mut sensors = builder.create_group("sensors");
# sensors.set_attr("location", AttrValue::AsciiString("lab_a".into()));
# sensors.create_dataset("pressure").with_f32_data(&[101.3, 101.5, 101.4]);
# sensors.create_dataset("humidity").with_f32_data(&[40.0, 41.5]);
# let mut imu = sensors.create_group("imu");
# imu.create_dataset("accel").with_f64_data(&[0.0, 0.0, 9.81]);
# sensors.add_group(imu.finish());
# builder.add_group(sensors.finish());
let file = File::from_bytes(builder.finish()?)?;

let root = file.root();
let root_attrs = root.attrs()?; // HashMap<String, AttrValue>

let sensors = file.group("sensors")?;
println!("child groups: {:?}", sensors.groups()?);   // ["imu"]
println!("datasets:     {:?}", sensors.datasets()?); // ["humidity", "pressure"]
println!("attributes:   {:?}", sensors.attrs()?);
# assert!(root_attrs.is_empty());
# assert_eq!(sensors.groups()?, vec!["imu"]);
# let mut names = sensors.datasets()?;
# names.sort();
# assert_eq!(names, vec!["humidity", "pressure"]);
# assert_eq!(sensors.attrs()?["location"], AttrValue::AsciiString("lab_a".into()));
# Ok::<(), hdf5_pure::Error>(())
```

[`File::group(path)`](crate::File::group) resolves a group by path, and [`Group::group(name)`](crate::Group::group) resolves a child relative to that group. Both fail with [`Error::NotAGroup`](crate::Error::NotAGroup) when the name reaches a dataset or a committed datatype, the way [`Error::NotADataset`](crate::Error::NotADataset) reports the mirror case for [`dataset()`](crate::Group::dataset). A name that reaches nothing at all is a [`FormatError::PathNotFound`](crate::FormatError::PathNotFound). The same error reports a component *along* a path, since resolving `a/b/c` opens `a` and then `a/b` to look inside them, and it states that component's own path: a dataset at `a/b` gives `Error::NotAGroup("a/b")` from `File::group("a/b/c")` and from `File::dataset("a/b/c")` alike. The names returned by [`groups()`](crate::Group::groups) and [`datasets()`](crate::Group::datasets) come in no guaranteed order, so sort them yourself if you need a stable listing.

### Reading an attribute value

An attribute reads back as the variant it was written from: the dataspace kind distinguishes a scalar from a one-element array, the datatype's charset selects the `Ascii` variants, and a number's width selects among [`I8`](crate::AttrValue::I8) … [`U64`](crate::AttrValue::U64) and [`F32`](crate::AttrValue::F32) / [`F64`](crate::AttrValue::F64). A [`VarLenAsciiCharArray`](crate::AttrValue::VarLenAsciiCharArray) of one element stays a [`VarLenAsciiCharArray`](crate::AttrValue::VarLenAsciiCharArray), an [`AsciiString`](crate::AttrValue::AsciiString) does not arrive as a [`String`](crate::AttrValue::String), and a 16-bit integer does not arrive as an [`I64`](crate::AttrValue::I64). A variable-length string is not a fixed-width one either: an h5py or C-library attribute arrives as [`VarLenString`](crate::AttrValue::VarLenString) or [`VarLenAsciiString`](crate::AttrValue::VarLenAsciiString) (or their array forms), so setting the value back writes the datatype it was found in.

A fixed-width string carries its width the same way. [`AttrValue::ascii_string_sized("ok", 64)`](crate::AttrValue::ascii_string_sized) declares a 64-byte slot, `H5T_C_S1` plus `H5Tset_size(64)`. A slot whose width differs from the one its content implies reads back as [`AsciiStringSized`](crate::AttrValue::AsciiStringSized) (or [`StringSized`](crate::AttrValue::StringSized), or the array forms) carrying that width, so writing the value back reproduces the slot at its declared width. A slot sized exactly to its content reads back as the plain [`AsciiString`](crate::AttrValue::AsciiString), which writes the same bytes.

That fidelity means several variants can carry the same logical value, so match on the variant only when the encoding is what you care about. Otherwise use the accessors, each of which spans every variant that can hold its shape:

```rust
# use hdf5_pure::{AttrValue, File, FileBuilder};
# let mut builder = FileBuilder::new();
# builder.set_attr("MATLAB_class", AttrValue::AsciiString("struct".into()));
# builder.set_attr(
#     "MATLAB_fields",
#     AttrValue::VarLenAsciiCharArray(vec!["alpha".into(), "beta".into()]),
# );
# let file = File::from_bytes(builder.finish()?)?;
let attrs = file.root().attrs()?;

// Any single string, either charset, scalar or one-element array.
let class: Option<&str> = attrs.get("MATLAB_class").and_then(AttrValue::as_str);

// Every element, with a scalar reading as one element.
let fields: Option<&[String]> = attrs.get("MATLAB_fields").and_then(AttrValue::as_strings);
# assert_eq!(class, Some("struct"));
# assert_eq!(fields, Some(&["alpha".to_string(), "beta".to_string()][..]));
# Ok::<(), hdf5_pure::Error>(())
```

[`as_i64`](crate::AttrValue::as_i64), [`as_u64`](crate::AttrValue::as_u64), [`as_f64`](crate::AttrValue::as_f64), [`to_i64s`](crate::AttrValue::to_i64s), [`to_u64s`](crate::AttrValue::to_u64s) and [`to_f64s`](crate::AttrValue::to_f64s) do the same for numbers. The prefix states the cost: `as_*` borrows or copies, `to_*` allocates. [`as_i64`](crate::AttrValue::as_i64) widens the narrower integer variants and reports `None` for a value above `i64::MAX`, per element, so the same holds for [`to_i64s`](crate::AttrValue::to_i64s) at any length. The float accessors do not convert integers, so a caller that accepts either calls both.

What a read cannot recover, because [`AttrValue`](crate::AttrValue) has no way to express it. Each of these reads correctly, and each would come back differently if it were *rewritten from the value*, which is why [`repack`](crate::repack()) copies an attribute's encoding:

- **Byte order and precision.** Every numeric variant writes back little-endian at its full width, so a big-endian attribute, one storing fewer bits than its bytes hold, or a float laid out other than the way IEEE 754 lays it out, reads correctly but would be re-encoded in this crate's own layout.
- **Integer widths without a Rust counterpart.** A number keeps its width at 1, 2, 4 and 8 bytes. A width with no Rust integer of its own, such as 3 bytes, widens to 64-bit.
- **A scalar in MATLAB's sequence-of-one-byte-strings shape.** Its array form reads as [`VarLenAsciiCharArray`](crate::AttrValue::VarLenAsciiCharArray), the form MATLAB writes. A scalar in that shape has no variant and reads as [`AsciiString`](crate::AttrValue::AsciiString), which writes a fixed-width slot. A standard variable-length string keeps its datatype at either arity.
- **Rank.** Every array variant is one-dimensional, so a rank-2 attribute reads as its elements flattened.
- **String padding.** A fixed-width string reports its content and the `STRSIZE` it was stored at, but not which padding rule filled the bytes past the content: every variant writes back `NULLPAD`.
- **Enumeration member names.** An enum attribute decodes through its integer base, so its codes arrive and its labels do not.

The variant may become **more specific** in a future release as [`AttrValue`](crate::AttrValue) grows further variants, as the numeric widths did, so match with a `_` arm, which the `#[non_exhaustive]` enum requires anyway, or read through the accessors, which are unaffected by such a change.

One case drops the value outright: a number **wider than 8 bytes** has no [`AttrValue`](crate::AttrValue) at all. The typed numeric readers model an element as a 64-bit word taken from its leading eight bytes, so a 9-byte integer holding 2^64 would read back as `0`, the same as one holding zero, and big-endian it would read back as 2^56, since those leading bytes are the *most* significant ones. [`attrs()`](crate::Group::attrs) omits such an attribute. [`attr_datatypes()`](crate::Group::attr_datatypes) still reports its full width, which separates it from a missing attribute.

### Reading an attribute's datatype

[`attr_datatypes()`](crate::Group::attr_datatypes) reports the on-disk [`Datatype`](crate::Datatype) of every attribute, keyed by name. It is the type channel to [`attrs()`](crate::Group::attrs)'s value channel, the pair a dataset already has in [`datatype()`](crate::Dataset::datatype) and its `read_*` methods, and it is where the *datatype* entries in the list above (byte order and precision, string padding and declared width, enumeration member names) can still be read.

**Rank is not among them.** An attribute's rank lives in its dataspace, which nothing public exposes, so [`attrs()`](crate::Group::attrs) delivers a rank-2 attribute flattened, and neither channel recovers its shape.

It reports **every** attribute message, including the ones [`attrs()`](crate::Group::attrs) omits because no [`AttrValue`](crate::AttrValue) can carry them, so a name that appears here and not in that map belongs to an attribute whose value has no variant.

A **committed** (shared) type, created with `H5Tcommit`, is stored on the attribute as a reference to the type's own object header, and both this and [`Dataset::datatype()`](crate::Dataset::datatype) follow that reference and report the type stored there. netCDF-4 user-defined types and h5py's `f["t"] = np.dtype(...)` reach a file this way. What neither channel reports is the type's *name*: two attributes sharing `/mytype` give the same [`Datatype`](crate::Datatype) as one that spells it out inline.

A boolean attribute needs both channels. The C library gives `H5T_NATIVE_HBOOL`, what h5py writes for every `np.bool_`, an `enum[FALSE, TRUE]` over an 8-bit base, so the value arrives as `0` or `1`, the same as an `i8`, and only the datatype records which it was. No [`AttrValue`](crate::AttrValue) variant writes an enumeration, so the example below writes an `i8`, which the datatype check separates from a boolean:

```rust
# use hdf5_pure::{AttrValue, File, FileBuilder};
# let mut builder = FileBuilder::new();
# builder.set_attr("success", AttrValue::I8(1));
# let file = File::from_bytes(builder.finish()?)?;
use hdf5_pure::Datatype;

let root = file.root();
let is_bool = matches!(
    root.attr_datatypes()?.get("success"),
    Some(Datatype::Enumeration { base_type, members, .. })
        if matches!(**base_type, Datatype::FixedPoint { size: 1, .. })
            && members.iter().any(|m| m.name == "TRUE")
);
let value = root.attrs()?.get("success").and_then(AttrValue::as_i64);
# assert!(!is_bool); // this crate writes `I8`, which the check separates from a boolean
# assert_eq!(value, Some(1));
# Ok::<(), hdf5_pure::Error>(())
```

### Addressing datasets

Datasets are addressable two ways: by full path from the file, or by name from their parent group. Both resolve to the same dataset.

```rust
# use hdf5_pure::{File, FileBuilder};
# let mut builder = FileBuilder::new();
# let mut sensors = builder.create_group("sensors");
# let mut imu = sensors.create_group("imu");
# imu.create_dataset("accel").with_f64_data(&[0.0, 0.0, 9.81]);
# sensors.add_group(imu.finish());
# builder.add_group(sensors.finish());
# let file = File::from_bytes(builder.finish()?)?;
// Full path from the file.
let accel = file.dataset("sensors/imu/accel")?;
println!("{:?}", accel.read_f64()?); // [0.0, 0.0, 9.81]

// By name, relative to the parent group.
let imu = file.group("sensors/imu")?;
let accel = imu.dataset("accel")?;
# assert_eq!(accel.read_f64()?, vec![0.0, 0.0, 9.81]);
# Ok::<(), hdf5_pure::Error>(())
```

See [Reading files](crate::_guide::reading) for the dataset read API ([`read_f64`](crate::Dataset::read_f64), [`shape`](crate::Dataset::shape), and friends).

## MATLAB struct convention

The ASCII attribute variants exist primarily so that groups can follow MATLAB's struct convention: a struct is a group carrying `MATLAB_class = "struct"` and a `MATLAB_fields` attribute (typically [`AttrValue::VarLenAsciiCharArray`](crate::AttrValue::VarLenAsciiCharArray)) listing the field names, with each field stored as a child dataset tagged with its own `MATLAB_class`. The MATLAB interop page has the full convention and worked examples.
