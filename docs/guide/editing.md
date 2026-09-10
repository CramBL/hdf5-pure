[`File::open_rw`](crate::File::open_rw) opens an existing HDF5 file for reading **and** writing, and edits it through owned [`Dataset`](crate::Dataset) and [`Group`](crate::Group) handles that reach every object by name: adding, copying, or deleting objects, or editing attributes, without rewriting the file from scratch. New data and rebuilt object headers are appended at the end of the file and the superblock is repointed last, so the cost is proportional to what changes and independent of the file size, and a failed commit leaves the original file valid.

This page mirrors [`examples/edit_in_place.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/edit_in_place.rs). Run it with:

```console
$ cargo run --example edit_in_place
```

## Choosing a write path

[`File::open_rw`](crate::File::open_rw) is the read-write open, and it picks how to hold the file's bytes from the file itself. A latest-format file with no userblock is edited **bounded**: no whole-file copy is ever built, so memory stays at the [configured caches](crate::_guide::streaming) plus whatever an edit is building. Anything else - a pre-v2 superblock, or a userblock - falls back to a whole-file **mirror**, `O(file size)`, which is what makes such a file editable at all. Those two are the whole fallback set: neither the file's size nor its file-space strategy enters into it, so a persisting [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) file of any size is edited bounded. Either backing mutates the file the same way: new bytes are appended and a small, fixed set of locations is patched, never a rewrite on commit. See the [crate documentation](crate) for the mechanics.

[`File::edit_backing()`](crate::File::edit_backing) reports the backing a file got, and [`FileAccessProperties::with_memory_strategy`](crate::FileAccessProperties::with_memory_strategy) demands one:

| [`MemoryStrategy`](crate::MemoryStrategy) | Effect |
| --- | --- |
| unset (the default) | Prefer bounded, and fall back to the mirror for a file the bounded engine cannot edit |
| [`Bounded`](crate::MemoryStrategy::Bounded) | Never build a mirror, and reject such a file with [`Error::EditUnsupported`](crate::Error::EditUnsupported) |
| [`Mirrored`](crate::MemoryStrategy::Mirrored) | Always build the mirror, whatever the file looks like |

[`File::edit_backing()`](crate::File::edit_backing) reports an [`EditBacking`](crate::EditBacking), [`Bounded`](crate::EditBacking::Bounded) or [`Mirrored`](crate::EditBacking::Mirrored), where the caller supplied a [`MemoryStrategy`](crate::MemoryStrategy): [`Auto`](crate::MemoryStrategy::Auto) is a preference between the two backings and never an outcome. `.into()` converts an [`EditBacking`](crate::EditBacking) back into the [`MemoryStrategy`](crate::MemoryStrategy) that pins a later reopen to it.

[`File::open_swmr_writer`](crate::File::open_swmr_writer) always mirrors, so it accepts [`Auto`](crate::MemoryStrategy::Auto) and [`Mirrored`](crate::MemoryStrategy::Mirrored), both of which the mirror satisfies, and rejects an explicit [`Bounded`](crate::MemoryStrategy::Bounded) at the open.

The two backings are the same engine and the same edit vocabulary, reads, immediate [`Dataset::append`](crate::Dataset::append) / [`append_raw`](crate::Dataset::append_raw), buffered [`Dataset::buffered_appender`](crate::Dataset::buffered_appender), [`Dataset::write`](crate::Dataset::write), [`append_staged`](crate::Dataset::append_staged), [`set_attr`](crate::Dataset::set_attr) / [`remove_attr`](crate::Dataset::remove_attr), [`Group::create_dataset`](crate::Group::create_dataset) / [`create_group`](crate::Group::create_group) / [`create_group_with`](crate::Group::create_group_with) / [`delete`](crate::Group::delete), [`File::copy`](crate::File::copy) / [`copy_from`](crate::File::copy_from), [`commit`](crate::File::commit), and [`space_accounting`](crate::File::space_accounting), with one trade between them. A large [`Dataset::append`](crate::Dataset::append) is one crash-atomic apply on the mirror and several ~1 MiB whole-chunk batches when bounded, so a crash mid-call there leaves a valid shorter dataset. A commit's resident memory follows the backing too: the edit bounds it, and the file's size does not, with [`File::copy`](crate::File::copy) the exception, since copying an object reads the whole of it into memory first.

One caveat inside either backing: the immediate [`Dataset::append`](crate::Dataset::append) is stricter than the staged surface. It needs a latest-format (v2/v3-superblock) file with no userblock, and the error it returns ([`Error::AppendInPlaceUnsupported`](crate::Error::AppendInPlaceUnsupported)) points at [`Dataset::append_staged`](crate::Dataset::append_staged) as the fallback.

The file's file-space strategy gates what an edit can do at all, whichever backing holds it:

| File-space strategy | Staged edits + [`commit`](crate::File::commit) | Immediate append |
| --- | --- | --- |
| None recorded, or [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) / [`Aggr`](crate::FileSpaceStrategy::Aggr) / [`None`](crate::FileSpaceStrategy::None) with `persist = false` | Yes | Yes |
| [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) / [`Aggr`](crate::FileSpaceStrategy::Aggr) / [`None`](crate::FileSpaceStrategy::None) with `persist = true` | Yes: freed space is recorded on disk | Yes: managers rewritten at [`close`](crate::File::close) |
| [`Page`](crate::FileSpaceStrategy::Page) with `persist = true` | Yes: page-aware commit, per-page-type managers rewritten | Yes: appends stay page-homogeneous |
| [`Page`](crate::FileSpaceStrategy::Page) with `persist = false` | No: rejected at open | No: rejected at open |

The one rejection here is [`Error::EditUnsupported`](crate::Error::EditUnsupported) for a paged file that does not persist its free space: neither backing can keep such a file's pages segregated, so [`open_rw`](crate::File::open_rw) rejects it at the open, before any byte of the file changes, and stages nothing against it. ([`MemoryStrategy::Mirrored`](crate::MemoryStrategy::Mirrored) still opens the file, since it demands a backing where the default expresses a preference, but a commit through it is rejected.) A [`Page`](crate::FileSpaceStrategy::Page) file with `persist = false` stays fully readable through every read path, and [`repack`](crate::repack()) rewrites it compactly.

A paged commit keeps each page homogeneous, so raw data and file metadata never share a page, apart from a chunked dataset's index, which travels with its chunk data. It also page-aligns the end of allocation, so the reference C library reopens the result as a paged file and recovers its free space. A free hole belongs to the page type of the page it sits in, and an allocation draws only on holes of its own type. A page holding nothing at all belongs to neither, so either type may open one. The page types are set out in the [file-space strategy](crate::_guide::file_space) guide.

For a brand-new file, use [`FileBuilder`](crate::FileBuilder), which [Writing files](crate::_guide::writing) walks. To append while readers are live, use the [SWMR writer](crate::_guide::swmr). To compact a file or drop objects across a reopen, use [`repack`](crate::repack()). The file properties reference lists the `fcpl` and `fapl` properties each path supports.

## Staging and committing edits

An edit session is transactional: you stage operations on an open file, then apply them all at once with [`commit()`](crate::File::commit). Nothing on disk changes until [`commit()`](crate::File::commit) succeeds.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("temperature").with_f64_data(&[22.5, 23.1, 21.8]);
# let mut sensors = builder.create_group("sensors");
# sensors.create_dataset("pressure").with_f32_data(&[101.3]);
# builder.add_group(sensors.finish());
# builder.write(&path)?;
use hdf5_pure::{AttrValue, File};

let file = File::open_rw(&path)?;
let root = file.root();

root.create_group_with("run2", |g| {
    g.set_attr("kind", AttrValue::AsciiString("trial".into()));
})?;
root.create_dataset("run2/signal", |b| {
    b.with_f64_data(&[1.0, 2.0, 3.0]);
})?;
file.copy("temperature", "temperature_backup")?; // H5Ocopy
root.delete("sensors/pressure")?;                // H5Ldelete

file.commit()?; // apply everything in place
# drop(root);
# drop(file);
# let reopened = File::open(&path)?;
# assert_eq!(reopened.dataset("run2/signal")?.read_f64()?, vec![1.0, 2.0, 3.0]);
# Ok::<(), hdf5_pure::Error>(())
```

After a successful [`commit()`](crate::File::commit), the staged set is cleared and the open file can be reused for further edits.

### Handles across a commit

[`Dataset`](crate::Dataset) and [`Group`](crate::Group) handles stay usable across [`commit()`](crate::File::commit). A commit relocates object headers, and each handle looks its object up again by path on its next use, so a handle held in a struct or a cache reads the file the commit left, never the copy it moved away from. An edit made through *another* handle to the same object reaches it the same way. Both types are `Clone`.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# let mut run2 = builder.create_group("run2");
# run2.create_dataset("signal").with_f64_data(&[1.0, 2.0, 3.0]);
# builder.add_group(run2.finish());
# builder.write(&path)?;
use hdf5_pure::{AttrValue, File};

let file = File::open_rw(&path)?;
let mut signal = file.dataset("run2/signal")?;

signal.set_attr("units", AttrValue::AsciiString("V".into()))?;
file.commit()?;

assert_eq!(signal.attrs()?["units"], AttrValue::AsciiString("V".into())); // same handle, new file
# Ok::<(), hdf5_pure::Error>(())
```

Three cases return an error:

- **Reading** through a handle onto an object the commit **deleted** fails the way opening it by name would ([`FormatError::PathNotFound`](crate::FormatError::PathNotFound)). Onto one the commit **replaced with an object of a different kind**, it is [`Error::NotADataset`](crate::Error::NotADataset) or [`Error::NotAGroup`](crate::Error::NotAGroup). The bytes a deleted object leaves behind still parse as the object that left them, so reading them would return data the file no longer holds. Its write methods address the file by path, so they still stage, and the commit rejects them.
- A handle reached by **object reference** ([`Dataset::dereference`](crate::Dataset::dereference)) holds an object-header address and no path to look up, so it returns [`Error::StaleHandle`](crate::Error::StaleHandle) once anything that could have moved that header has run: a commit, a staged edit, a [`sync`](crate::File::sync), a [`close`](crate::File::close). Dereference again from a fresh read. An immediate [`append`](crate::Dataset::append) moves no header, so such a handle keeps reading and appending across one.
- A handle onto an object the session has **staged** and not yet committed returns [`Error::NotCommitted`](crate::Error::NotCommitted) for anything that reads bytes, and serves everything else. The commit is what turns it into an ordinary handle. See [Staged objects are addressable straight away](#staged-objects-are-addressable-straight-away).

## Operations

| Method | Effect | HDF5 analog |
| --- | --- | --- |
| [`File::open_rw(path)`](crate::File::open_rw) | Open an existing file for reading and writing | none |
| [`Group::create_group(name)`](crate::Group::create_group) | Stage a new empty group, returning a [`Group`](crate::Group) handle onto it | `H5Gcreate` |
| [`Group::create_group_with(name, build)`](crate::Group::create_group_with) | Stage a new group, configured through `build` (attributes, nested objects), returning a [`Group`](crate::Group) handle onto it | `H5Gcreate` |
| [`Group::create_dataset(name, build)`](crate::Group::create_dataset) | Stage a new dataset, configured through a [`DatasetBuilder`](crate::DatasetBuilder), returning a [`Dataset`](crate::Dataset) handle onto it | `H5Dcreate` |
| [`Dataset::append_staged(build)`](crate::Dataset::append_staged) | Stage appending elements along axis 0 of an existing chunked, unlimited dataset, via an [`AppendBuilder`](crate::AppendBuilder) | `H5Dset_extent` + write |
| [`Dataset::append(data)`](crate::Dataset::append) | Append immediately and durably, no [`commit`](crate::File::commit) needed | `H5Dset_extent` + write |
| [`Dataset::buffered_appender()`](crate::Dataset::buffered_appender) | Buffer appended elements and write them a whole chunk at a time | `H5Pset_chunk_cache` (write side) |
| [`Dataset::write(data)`](crate::Dataset::write) / [`write_staged(build)`](crate::Dataset::write_staged) | Stage a value overwrite of the same datatype and shape | `H5Dwrite` |
| [`Group::set_attr(name, value)`](crate::Group::set_attr) / [`Dataset::set_attr`](crate::Dataset::set_attr) | Stage adding or replacing a compact attribute | none |
| [`Group::remove_attr(name)`](crate::Group::remove_attr) / [`Dataset::remove_attr`](crate::Dataset::remove_attr) | Stage removing a compact attribute | none |
| [`File::copy(src, dst)`](crate::File::copy) | Stage a deep copy of a dataset or whole group subtree within this file | `H5Ocopy` |
| [`File::copy_from(source, src, dst)`](crate::File::copy_from) | Copy a dataset or subtree out of another open [`File`](crate::File) into this one | `H5Ocopy` (across files) |
| [`Group::delete(name)`](crate::Group::delete) | Stage removing the link at `name` (and, for a group, its whole subtree) | `H5Ldelete` |
| [`File::commit()`](crate::File::commit) | Apply all staged operations in place and flush | none |

[`create_dataset`](crate::Group::create_dataset) hands you the same [`DatasetBuilder`](crate::DatasetBuilder) used by [`FileBuilder`](crate::FileBuilder), so you configure the new dataset exactly as you would when creating a file from scratch:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
# hdf5_pure::FileBuilder::new().write(&path)?;
# let file = hdf5_pure::File::open_rw(&path)?;
# let root = file.root();
# root.create_group("run2")?;
root.create_dataset("run2/signal", |b| {
    b.with_f64_data(&[1.0, 2.0, 3.0]);
})?;
# file.commit()?;
# Ok::<(), hdf5_pure::Error>(())
```

A new dataset may be created **empty**, with zero elements, chunked, with or without an unlimited maximum, which is how a schema-first writer declares its columns before any data has arrived and then grows each one with [`append_staged`](#appending-to-an-unlimited-dataset) as batches come in:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("schema.h5");
# hdf5_pure::FileBuilder::new().write(&path)?;
# let file = hdf5_pure::File::open_rw(&path)?;
# let root = file.root();
root.create_dataset("col", |b| {
    b.with_f64_data(&[])
        .with_shape(&[0])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[512]);
})?;
# file.commit()?;
# Ok::<(), hdf5_pure::Error>(())
```

The chunk dimensions have to be given: auto-chunking derives them from the shape, and a zero-element shape has none to derive from.

The [`create_group_with`](crate::Group::create_group_with) / [`create_dataset`](crate::Group::create_dataset) / [`write_staged`](crate::Dataset::write_staged) / [`append_staged`](crate::Dataset::append_staged) closures configure a builder and leave the file itself alone, so a closure may read the same [`File`](crate::File): staging a dataset whose contents depend on one already there works. What it reads is the file as it was before the call, since the closure runs before its edits are recorded.

[`set_attr`](crate::Group::set_attr) takes an [`AttrValue`](crate::AttrValue), fixed-size or variable-length. [`File::root()`](crate::File::root) returns the root group. Attributes go in the rebuilt header while they fit it compactly. Past that, at more than eight of them, or at one too large for an object-header message, the whole set moves to a fractal heap, and an object already storing its attributes in one is rebuilt the same way ([#102](https://github.com/CramBL/hdf5-pure/issues/102)). An object stays dense once it is dense, and the heap goes away only when its last attribute is removed. A rebuilt heap re-encodes the attributes it carries over: name, datatype, dataspace and value survive (the name's declared character set is rewritten as ASCII), and the heap the edit replaces is left as dead bytes for [`repack`](crate::repack()) to recover. An object that tracks attribute creation order keeps it across the rebuild, creation-order index and all ([#416](https://github.com/CramBL/hdf5-pure/issues/416)). A dataset or group *created* in place may carry a dense set the same way.

An object that tracks **attribute creation order**, which is h5py's `track_order=True`, `H5Pset_attr_creation_order`, and everything netCDF-4 writes, is edited like any other ([#416](https://github.com/CramBL/hdf5-pure/issues/416)). A new attribute takes the object's next creation index, overwriting one keeps the index it had, and deleting one leaves a gap and renumbers nothing, so an iteration by creation order reads the same afterwards. A group that tracks **link** creation order works the same way: a dataset or group created inside it takes the next link creation index (several added in one commit take consecutive ones, in the order they are placed), a deletion leaves a gap without lowering the group's counter, and copying the group keeps the order it had. A group *created* that way is a plain group and does not itself track the order, since the tracking is a property of the group whose links are being numbered. The one rejection left is an addition that would take such a group past the compact-storage threshold its group-info message declares, 8 links by default, since its links would then move to a fractal heap indexed by a creation-order B-tree this crate does not write ([#102](https://github.com/CramBL/hdf5-pure/issues/102)). A group whose links are already dense is rejected for the same reason, tracked or not.

Rebuilding a version 2 object header preserves the two optional blocks its prefix may carry: the four **timestamps** (access, modification, change, birth) that the reference C library stores on every header it writes, and the **attribute phase-change thresholds** set with `H5Pset_attr_phase_change` ([#422](https://github.com/CramBL/hdf5-pure/pull/422)). The modification and change times are moved to the time of the edit, since a rewrite is a modification. The access and birth times are copied as they were, and a `no_std` build, having no clock, copies all four. A header this crate creates from nothing leaves the timestamp block out, so its whole-file output stays byte-for-byte reproducible. The thresholds are preserved and left unread: the editor's own compact-to-dense switch is the fixed eight attributes that the limitations reference describes under dense attribute storage.

[`copy`](crate::File::copy) performs a deep copy: fresh copies of every object's data and header are written, internal links and the contiguous data address are repointed to the copies, and a link named by the last component of `dst` is added to its parent group. The original is untouched. `src` must exist and `dst` must not (and may not lie inside `src`). Compact attributes are carried over byte-for-byte, including the latest-format form the C library and h5py write, where a handful of inline attributes are accompanied by an Attribute Info message. Dense (fractal-heap) attribute storage, which an object takes on above 8 attributes or when a single attribute is too large for an object-header message, is also reproduced: the source attributes are read out of the source heap and re-emitted into a fresh single-direct-block fractal heap plus B-tree v2 name index in the destination, with a creation-order index beside it where the source object declared one. An attribute too large even for a managed heap object is re-emitted as a *huge* object, as it was in the source. More attributes than the single B-tree leaf can index is rejected, and the attribute is named in the error, never mis-encoded. The set's *total* size has no limit, since the heap's one block is sized to the content.

[`copy_from`](crate::File::copy_from) is the same operation **across two open files**, the cross-file form of `H5Ocopy`. The source lives in a separate [`File`](crate::File) reader, apart from the file being edited:

```rust
# let dir = tempfile::tempdir()?;
# let library_path = dir.path().join("library.h5");
# let path = dir.path().join("output.h5");
# let mut source = hdf5_pure::FileBuilder::new();
# source.create_dataset("calibration").with_f64_data(&[0.5, 1.5]);
# source.write(&library_path)?;
# let mut builder = hdf5_pure::FileBuilder::new();
# let mut run2 = builder.create_group("run2");
# run2.create_dataset("signal").with_f64_data(&[1.0]);
# builder.add_group(run2.finish());
# builder.write(&path)?;
use hdf5_pure::File;

let library = File::open(&library_path)?;
let file = File::open_rw(&path)?;
file.copy_from(&library, "calibration", "run2/calibration")?;
file.commit()?;
# drop(file);
# assert_eq!(File::open(&path)?.dataset("run2/calibration")?.read_f64()?, vec![0.5, 1.5]);
# Ok::<(), hdf5_pure::Error>(())
```

Unlike [`copy`](crate::File::copy), the source subtree is read and validated **eagerly** (the [`File`](crate::File) borrow need not outlive the call), so [`copy_from`](crate::File::copy_from) returns a `Result`, and the destination still changes only on [`commit()`](crate::File::commit). Because the copy is byte-for-byte verbatim, anything whose stored bytes embed a *source-file* absolute address, which would dangle in another file, is rejected up front: variable-length and reference datasets and attributes (whether compact or dense), and any shared header message (a committed datatype, or an SOHM-shared dataspace, fill value, or filter pipeline). The same-file [`copy`](crate::File::copy) keeps these forms valid instead, by sharing the source file's global heaps and objects. The `source` must be a buffered file ([`File::open`](crate::File::open) or [`File::from_bytes`](crate::File::from_bytes), and not [`File::open_streaming`](crate::File::open_streaming)) using 8-byte offsets and no userblock.

### Replacing an object

Deleting a path and creating something new at that same path in one commit replaces it. The removal is applied before the addition, and the commit's single superblock write publishes both, so a rotating store, a ring buffer of tables or a rolling window of daily datasets, expresses a rotation as one commit, and the path is never momentarily absent:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("ring.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("t0").with_i32_data(&[0]);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open_rw(&path)?;
let root = file.root();

root.delete("t0")?;
root.create_dataset("t0", |b| { b.with_i32_data(&[1, 2, 3]); })?;
file.commit()?;
# drop(root);
# drop(file);
# assert_eq!(File::open(&path)?.dataset("t0")?.read_i32()?, vec![1, 2, 3]);
# Ok::<(), hdf5_pure::Error>(())
```

The replacement need not resemble the original: a dataset may replace a group or the reverse, and replacing a group discards its whole subtree.

Everything the commit adds *below* a replaced path lands in the replacement, whatever order the calls were made in: staging [`create_dataset("g/x")`](crate::Group::create_dataset) and then replacing `g` puts `x` in the new group, never in the one being removed. What is rejected is a staged edit that could only mean the *original*:

| staged beside [`delete("g")`](crate::Group::delete) | |
| --- | --- |
| [`create_dataset("g/x")`](crate::Group::create_dataset) with no replacement of `g` | rejected: it would add into a group whose own link the commit removes |
| a group under `g` that the commit does not itself create | rejected |
| an attribute set on `g`, or a value overwrite inside it | rejected |
| [`copy("g/inner", "backup")`](crate::File::copy), a copy reading from the replaced subtree | rejected: a copy takes its bytes from the pre-commit file, so this would place the original at `backup` while the replacement lands at `g` |

A source that is deleted but *not* replaced is unambiguous, since that is a move, and stays allowed.

The new object's storage is appended, and the original's bytes stay where they are until the superblock is repointed, which is what makes a crash mid-rotation land on one side or the other. The space the deletion released is therefore what a *later* commit draws on, by the two mechanisms [Space reuse and truncation](#space-reuse-and-truncation) describes: reuse for an interior region, truncation for one that reaches the end of the file. A rotation loop in one session reaches a steady file size.

## Staged objects are addressable straight away

[`create_group`](crate::Group::create_group), [`create_group_with`](crate::Group::create_group_with) and [`create_dataset`](crate::Group::create_dataset) each return a handle onto the object they stage, and [`File::group`](crate::File::group) / [`File::dataset`](crate::File::dataset) / [`Group::group`](crate::Group::group) / [`Group::dataset`](crate::Group::dataset) find it by name from the same session. A nested schema is therefore built and its handles cached in one pass, with no commit in the middle:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("runs.h5");
# hdf5_pure::FileBuilder::new().write(&path)?;
use hdf5_pure::File;

let file = File::open_rw(&path)?;
let epoch = file.root().create_group_with("epoch", |g| {
    g.create_dataset("sys_time", |b| {
        b.with_u64_data(&[])
            .with_shape(&[0])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[512]);
    });
})?;

let mut sys_time = epoch.dataset("sys_time")?; // before the commit
assert_eq!(sys_time.shape()?, vec![0]);
sys_time.append_staged(|a| { a.append_u64(&[1, 2, 3]); })?;

file.commit()?;
assert_eq!(sys_time.read_u64()?, vec![1, 2, 3]); // the same handle
# Ok::<(), hdf5_pure::Error>(())
```

Such a handle *addresses* the object: it cannot read bytes the file has yet to hold. What it does before the commit:

| Through a handle onto a staged object | Before [`commit`](crate::File::commit) |
| --- | --- |
| [`Group::create_group`](crate::Group::create_group) / [`create_group_with`](crate::Group::create_group_with) / [`create_dataset`](crate::Group::create_dataset) / [`delete`](crate::Group::delete) / [`set_attr`](crate::Group::set_attr) / [`remove_attr`](crate::Group::remove_attr) | stages, exactly as on a committed group |
| [`Group::group`](crate::Group::group) / [`dataset`](crate::Group::dataset), [`groups()`](crate::Group::groups) / [`datasets()`](crate::Group::datasets) / [`iter_groups()`](crate::Group::iter_groups) / [`iter_datasets()`](crate::Group::iter_datasets) | the staged members |
| [`Dataset::shape`](crate::Dataset::shape) / [`maxshape`](crate::Dataset::maxshape) / [`dtype`](crate::Dataset::dtype) / [`datatype`](crate::Dataset::datatype) / [`is_chunked`](crate::Dataset::is_chunked) / [`filters`](crate::Dataset::filters) | what the builder settled |
| [`Dataset::append_staged`](crate::Dataset::append_staged) | the elements join the pending creation |
| [`Dataset`](crate::Dataset) reads (`read_*`, [`attrs`](crate::Dataset::attrs)), [`Group::attrs`](crate::Group::attrs) | [`Error::NotCommitted`](crate::Error::NotCommitted) |
| [`Dataset::append`](crate::Dataset::append) / [`append_raw`](crate::Dataset::append_raw) / [`buffered_appender`](crate::Dataset::buffered_appender) | [`Error::NotCommitted`](crate::Error::NotCommitted): no bytes to grow in place |
| [`Dataset::write`](crate::Dataset::write) / [`write_staged`](crate::Dataset::write_staged) / [`set_attr`](crate::Dataset::set_attr) / [`remove_attr`](crate::Dataset::remove_attr) | [`Error::NotCommitted`](crate::Error::NotCommitted) |

[`append_staged`](crate::Dataset::append_staged) on a staged dataset is the same edit as having handed the elements to its builder, so it needs neither an unlimited dimension nor a chunked layout, the two an append onto a *committed* dataset does need. A variable-length-string or object-reference dataset is the exception: its per-element side table is built with the dataset, so supply every element through the builder.

After [`commit`](crate::File::commit) the same handles read and write the objects in the file. A rejected commit leaves them staged and still returning [`NotCommitted`](crate::Error::NotCommitted), and the same batch can be committed again.

These handles are ordinary [`Group`](crate::Group) and [`Dataset`](crate::Dataset) handles, so each **holds the file's exclusive OS lock alive**: `let ds = root.create_dataset(..)?` holds the lock for as long as `ds` lives, where `root.create_dataset(..)?;` drops it at the semicolon. Reopening the file (mandatory on Windows, where the lock is not advisory) fails until the handle is dropped. See [`File::close`](crate::File::close) for the rule.

Only a path the session **named** is addressable this way, and every group on the way to an addition has to be in the file already or staged in the same session. [`create_dataset("a/b")`](crate::Group::create_dataset) with no `a` anywhere stages, and the commit rejects the whole batch with [`Error::EditUnsupported`](crate::Error::EditUnsupported) ([#533](https://github.com/CramBL/hdf5-pure/issues/533)). Stage the group by name ([`create_group("a")`](crate::Group::create_group)) before the dataset, which gives the commit its parent and hands you a handle on it.

Two more rules follow from what [`commit`](crate::File::commit) itself does. A creation whose name **collides** with a link the commit keeps is rejected *where it is staged*, before the commit runs, because there would be no new object for the handle to address: [`create_dataset("existing")`](crate::Group::create_dataset) returns [`Error::EditUnsupported`](crate::Error::EditUnsupported) unless the same session also deletes that link. A second creation at a path the session **already staged** one at is rejected there too, since one name cannot mean two objects: the staged set is indexed by path, so the handle would report the first creation's shape and datatype as the second one's. Staging the same *group* twice is the exception and stays allowed, since both calls refer to one group, which is how attributes and children are added to a group already staged. And deleting something the session staged and has not committed **withdraws** that staging, its attributes, appends and staged children with it, since there is no link in the file to unlink. Where the file does hold a link at the path, the plain deletion of it is what remains. A handle onto a withdrawn creation reports [`Error::StagingWithdrawn`](crate::Error::StagingWithdrawn) from then on: it refers to nothing, and the object the file holds at that path is the one the session is deleting.

A deletion carries a group's whole subtree away, but only a commit that builds that group **again** puts anything back under it. So [`delete("g")`](crate::Group::delete) plus [`create_dataset("g/x")`](crate::Group::create_dataset) with no new `g` is a batch the commit rejects, and until then `g`'s own children still own their names, so reads and listings of them keep working. The root cannot be deleted.

[Replacing an object](#replacing-an-object) reads the same before the commit as after: the path refers to the replacement, and not to the object being removed. A deletion on its own hides nothing, since the object is in the file until the commit runs, and reads through it keep working.

[`StagedGroup`](crate::StagedGroup), which [`create_group_with`](crate::Group::create_group_with)'s closure receives, describes a whole subtree in one call, and a group that is not committed is reachable by name as well. Its methods return `&mut Self` for chaining and hand back no handles, so look the object up by name once the closure has returned.

## Appending to an unlimited dataset

[`Dataset::append_staged`](crate::Dataset::append_staged) grows an existing **chunked, unlimited** dataset in place along its first (axis-0) dimension, **including filtered** datasets (deflate, shuffle, fletcher32, scale-offset, LZF, and ZFP with the `zfp` feature). It is the general, non-SWMR counterpart to the SWMR writer, which appends only to *unfiltered*, chunk-aligned datasets. It returns an [`AppendBuilder`](crate::AppendBuilder) whose typed and generic methods mirror the writer's, and repeated calls concatenate in call order.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("log.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("samples")
#     .with_i32_data(&[5, 6, 7])
#     .with_maxshape(&[u64::MAX])
#     .with_chunks(&[4]);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open_rw(&path)?;
file.dataset("samples")?
    .append_staged(|a| {
        a.append_i32(&[8, 9, 10, 11]);
    })?;
file.commit()?;
# drop(file);
# assert_eq!(File::open(&path)?.dataset("samples")?.read_i32()?, vec![5, 6, 7, 8, 9, 10, 11]);
# Ok::<(), hdf5_pure::Error>(())
```

Existing chunks stay exactly where they are. Only the newly appended chunks are compressed and written, plus the single trailing partial chunk, when the dataset's current length is not a whole multiple of the chunk length. Every other chunk is carried into the rebuilt index by metadata alone, so an append does not rewrite existing data and the file does not grow by the whole dataset each time. Appends of any length are allowed, and the datatype, fill value, filter pipeline, and attributes are preserved.

Like every staged edit, an append commits by writing the new chunks and a rebuilt index at end-of-file and repointing the superblock last (under the file's exclusive lock), so a crash leaves either the original dataset or the fully grown one, never a torn state. It sets no SWMR flag.

### Eligibility

The engine supports the Extensible-Array chunk index, which the reference C library and h5py select for a single unlimited dimension under the latest format and this crate writes for every unlimited dataset, on rank-1 datasets that have a single hard link. A dataset that is not chunked, not unlimited along axis 0, not Extensible-Array indexed, higher than rank 1, uses a filter this engine cannot re-encode, or has a sparse chunk grid is rejected with [`Error::AppendUnsupported`](crate::Error::AppendUnsupported) before any file bytes change. Check eligibility up front with the read-side accessors [`is_chunked`](crate::Dataset::is_chunked), [`maxshape`](crate::Dataset::maxshape), [`chunk_shape`](crate::Dataset::chunk_shape), and [`filters`](crate::Dataset::filters), which report it without an attempted append (the [reading guide](crate::_guide::reading#chunking-filters-and-append-eligibility) has the four in one table).

Element types are checked, never coerced: each typed `append_*` call records the datatype it implies, and [`commit`](crate::File::commit) rejects a mismatch against the dataset's on-disk datatype, a mix of element types in one builder included. [`append_raw`](crate::Dataset::append_raw) appends already-little-endian element bytes verbatim. Its length must be a whole multiple of the element size, and the dataset's datatype must be little-endian.

This section mirrors [`examples/append_dataset.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/append_dataset.rs). Run it with `cargo run --example append_dataset`.

### Streaming appends

[`append_staged`](crate::Dataset::append_staged) rebuilds the dataset's chunk index and relocates its header on every [`commit`](crate::File::commit) (and each new [`File::open_rw`](crate::File::open_rw) re-reads the metadata it needs, or the whole file when it falls back to the mirror), which is the right trade for a one-off append composed alongside other edits, and the wrong one for a high-frequency append loop. For that, open the file **once** with [`File::open_rw`](crate::File::open_rw) and append many times through a [`Dataset`](crate::Dataset) handle, growing the Extensible-Array index *in place*: each append then costs `O(appended bytes)` plus amortized `O(1)` index overhead, with no whole-file re-read and no index rebuild.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("log.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("samples")
#     .with_i32_data(&[5, 6, 7])
#     .with_maxshape(&[u64::MAX])
#     .with_chunks(&[4]);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open_rw(&path)?;
let mut samples = file.dataset("samples")?;
samples.append(&[8i32, 9, 10, 11])?;
samples.append(&[12i32, 13])?; // any length, from any length
file.close()?;
# drop(samples);
# assert_eq!(File::open(&path)?.dataset("samples")?.read_i32()?.len(), 9);
# Ok::<(), hdf5_pure::Error>(())
```

One open file reaches every dataset by name, takes an exclusive file lock for its lifetime, and sets no SWMR flag. **Every [`append`](crate::Dataset::append) is crash-atomic**: writes are ordered child-before-parent with `fsync` barriers and the dataspace dimension is published last as the single commit point, so a crash between appends leaves either the previous length or the new one, and never a torn or lost view. Throughout this page, "crash-atomic" means that ordering holds against a process crash under either [`SyncPolicy`](crate::SyncPolicy) ([Choosing the fsync cadence](#choosing-the-fsync-cadence)), and against power loss under the default one.

An append may **start anywhere** under a lossless pipeline, filtered or not. When the current length is not chunk-aligned, the trailing partial chunk is rewritten into a *fresh* allocation, decoded, extended and re-encoded when the dataset is filtered, and its index element is repointed once the new bytes are on the disk. The old slot is never overwritten, so a crash mid-append leaves a reader at the old length decoding the new chunk, whose live prefix is byte-for-byte the one it replaced.

A **lossy** pipeline (ZFP, or float D-scale scale-offset) is the exception: re-encoding the trailing chunk does not reproduce the values it was decoded from, since a ZFP block re-quantizes to fit whichever values share it, so growing one on this path is rejected, and data that is already committed is never rewritten. Such a dataset takes whole-chunk appends from a chunk-aligned length. This is the same line [`repack`](crate::repack()) draws for its re-encoding paths, and [`Dataset::append_staged`](crate::Dataset::append_staged) draws it too.

The appended **length** is unconstrained too. An unaligned length only makes the last chunk the append writes a partial one, and that chunk's index element is a fresh insert past the old dimension, invisible until the dimension is published, exactly like every whole chunk beside it.

The SWMR writer is the exception: it keeps the chunk-aligned, unfiltered subset, because its readers are concurrent by contract and follow the chunk-index counts and not the dataspace dimension.

The remaining eligibility rules match [`Dataset::append_staged`](crate::Dataset::append_staged) (chunked, unlimited axis 0, Extensible-Array index, rank 1, a re-encodable filter pipeline), plus the file-level gates in [the tables above](#choosing-a-write-path), with one difference: it grows the index in place, so the index must already be allocated. This crate allocates it eagerly, so an empty dataset it wrote can be grown from the first append. An empty dataset the C library created without any initial data defers its index and is rejected, so make that first append with [`Dataset::append_staged`](crate::Dataset::append_staged), which materializes the index, or create the dataset with initial data. The dead bytes left when a partial trailing chunk is relocated are recovered by [`repack`](crate::repack()), and the session reuses none of them. This is the throughput-oriented counterpart to [`Dataset::append_staged`](crate::Dataset::append_staged) and the filter-capable counterpart to the SWMR writer.

This section mirrors [`examples/append_streaming.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/append_streaming.rs). Run it with `cargo run --example append_streaming`.

### Buffered appends

[`Dataset::append`](crate::Dataset::append) writes on every call: it encodes the appended elements, places their chunks, extends the index, and fsyncs five times (per batch, and under the default [`SyncPolicy`](crate::SyncPolicy)). That is the right trade for a caller appending a chunk at a time, and the wrong one for a caller appending a hundred elements at a time into a chunk that holds a thousand, which pays the whole sequence ten times over to write one chunk.

[`Dataset::buffered_appender`](crate::Dataset::buffered_appender) returns a [`BufferedAppender`](crate::BufferedAppender) that holds appended elements in memory and writes them only when they complete a chunk. It is this crate's equivalent of the reference C library's raw-data chunk cache, and it carries the same bargain: buffered elements are not in the file until the appender flushes.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("telemetry.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("samples")
#     .with_f64_data(&[])
#     .with_shape(&[0])
#     .with_maxshape(&[u64::MAX])
#     .with_chunks(&[1024]);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open_rw(&path)?;
let mut samples = file.dataset("samples")?;
let mut appender = samples.buffered_appender()?;
for batch in 0..1000 {
    appender.append(&[f64::from(batch); 100])?; // buffered; writes once a chunk fills
}
appender.finish()?;                             // the partial tail reaches the file here
# drop(samples);
# drop(file);
# assert_eq!(File::open(&path)?.dataset("samples")?.shape()?, vec![100_000]);
# Ok::<(), hdf5_pure::Error>(())
```

Every call that does not complete a chunk is a memory copy and nothing else. When one or more chunks are complete, exactly those chunks go through the immediate, crash-atomic in-place path and the remainder stays buffered, so a caller appending `k` elements at a time into a chunk of `n` writes once per `n/k` calls, where an unbuffered append writes once per call.

Each write the appender makes is itself crash-atomic, so a crash loses the buffered tail and never the file: the dataset reads back as the prefix that was written. [`flush`](crate::BufferedAppender::flush) publishes the buffered tail without consuming the appender, [`finish`](crate::BufferedAppender::finish) flushes and consumes it, and dropping without either still flushes, though a failure there cannot be reported, so prefer [`finish`](crate::BufferedAppender::finish) where the error matters. Eligibility is the same as [`Dataset::append`](crate::Dataset::append)'s and is reported when the appender is constructed, not on the first write.

A filtered dataset whose on-disk length is not a whole multiple of its chunk length, a log resumed across sessions typically, costs nothing extra: the in-place path re-encodes that trailing chunk into a fresh allocation. (A dataset under a *lossy* pipeline is the exception, and such an appender is rejected when it is constructed.) The appender picks its write prefix so the on-disk length lands back on a chunk boundary, which keeps the re-encoding to the first write alone. Flushing a partial chunk mid-stream leaves the length unaligned again, so let the appender batch where the last few elements can wait.

An appender holds elements the call that took them reported as accepted, and only its own flush can write them, and the immediate append path rejects a dataset with a staged edit on it or an ancestor. So while an appender is live the session **rejects that edit at the call that makes it**, with [`Error::EditUnsupported`](crate::Error::EditUnsupported), so the failure lands at the call and never at a flush a `Drop` could not report: any edit that reaches the dataset or an ancestor, and a second appender on the same dataset. An edit that reaches something else is accepted, since the appender stages nothing and has no commit of its own to protect. Prefer [`finish`](crate::BufferedAppender::finish) (or [`flush`](crate::BufferedAppender::flush)) over dropping anyway, since a drop cannot return an error. What stays outside the guarantee is [`File::close`](crate::File::close), which does not flush live appenders.

A **SWMR** writer requires the appended length to be chunk-aligned as well, so it can never write a partial trailing chunk. [`buffered_appender`](crate::Dataset::buffered_appender) rejects such a session outright, and never takes elements it could not flush.

Memory follows the buffer and not the file: an appender holds the unwritten elements plus a copy of the prefix it is writing, so peak is about twice the chunk size above whatever one call hands it. That is a different bound from the one [`Dataset::append`](crate::Dataset::append) gives ([Bounded-memory appends](#bounded-memory-appends)), which is independent of how much a single call appends.

This section mirrors [`examples/append_buffered.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/append_buffered.rs). Run it with `cargo run --example append_buffered`.

### Bounded-memory appends

Appending to a large file needs no special entry point: [`File::open_rw`](crate::File::open_rw) already edits a latest-format file bounded, the read-write sibling of [`open_streaming`](crate::File::open_streaming). Reads are served by positioned I/O with the streaming backend's capabilities, and [`Dataset::append`](crate::Dataset::append) runs the same crash-atomic in-place engine as the mirror, reading and patching only bounded windows (the object header, the extensible-array blocks it touches, and the trailing chunk). A large append is applied in whole-chunk batches, each crash-atomic on its own, so peak memory stays at the configured caches plus a few chunks, independent of the file size and of how much one call appends.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("huge-log.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("samples")
#     .with_i32_data(&[5, 6, 7])
#     .with_maxshape(&[u64::MAX])
#     .with_chunks(&[4]);
# builder.write(&path)?;
use hdf5_pure::{EditBacking, File, FileAccessProperties, MemoryStrategy};

// Bounded because the file allows it; add the hint to make it a requirement
// rather than a preference.
let file = File::open_rw_with_options(
    &path,
    FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
)?;
assert_eq!(file.edit_backing(), Some(EditBacking::Bounded));
let mut samples = file.dataset("samples")?;
samples.append(&[8i32, 9, 10, 11])?;
file.close()?;
# drop(samples);
# assert_eq!(File::open(&path)?.dataset("samples")?.read_i32()?.len(), 7);
# Ok::<(), hdf5_pure::Error>(())
```

Bounded editing **does** grow a file that persists its free space, a genuine paged file (`H5F_FSPACE_STRATEGY_PAGE`) included, seeding the on-disk free-space managers on open and rewriting them at [`File::close`](crate::File::close), with paged appends kept page-homogeneous (raw and metadata in separate pages). The [file-space strategy](crate::_guide::file_space) guide walks the paged details. Memory budgets are set with the same [`FileAccessProperties`](crate::FileAccessProperties) as the streaming reader, and cached metadata windows touched by an append are invalidated automatically, so reads through the same file never observe stale bytes.

The metadata cache is **off** by default, which is why a bounded session's resident memory starts near zero and not near the file's size. [`FileAccessProperties::with_metadata_cache(MetadataCacheConfig::new(bytes))`](crate::FileAccessProperties::with_metadata_cache) turns it on to pin the metadata a long append loop keeps re-reading, against a byte budget you choose and that never scales with the file's length. [`File::metadata_cache_stats`](crate::File::metadata_cache_stats) reports the hit rate and occupancy that say whether the budget was the right one.

## How it works

[`commit()`](crate::File::commit) appends each new dataset (its data blob and object header) and each new group, then appends rewritten object headers for every touched group and its ancestors up to the root (omitting any deleted links), and finally repoints the superblock at the new root.

The appended data is `fsync`ed before the root is repointed, so the "repoint last" guarantee is real: if the process or machine fails during a commit, the original file is still intact and readable, because the superblock still points at the old root. (Under [`SyncPolicy::OnClose`](crate::SyncPolicy::OnClose) the repoint is still last, and during the session only a *process* failure is ordered against. See [Choosing the fsync cadence](#choosing-the-fsync-cadence).) The cost of a commit scales with the size of the edit, not the size of the file.

**Every check runs before the first byte is written.** On any [`Error::EditUnsupported`](crate::Error::EditUnsupported), the file on disk is left untouched, so an unsupported edit is safe to attempt: it fails cleanly and leaves the file whole.

A rejection costs the session nothing it had staged, so nothing is lost by attempting a commit. A rejected [`commit()`](crate::File::commit) puts the staged set back whole: [`has_staged_edits()`](crate::File::has_staged_edits) still returns `true`, and committing again returns the same error, having applied no part of the batch. A staging call that returns an error stages nothing, a [`create_group_with`](crate::Group::create_group_with) whose closure had already recorded several edits included.

Which errors arrive where follows from that: a dataset is validated as it is staged, so a bad shape, a missing datatype or an unsupported combination is reported by [`create_dataset`](crate::Group::create_dataset) or [`write_staged`](crate::Dataset::write_staged) itself, while anything that has to read the file is reported by [`commit()`](crate::File::commit): a value overwrite that is not the on-disk datatype or shape, a deletion that overlaps another edit, a target that does not exist.

A value overwrite replaces element bytes and nothing else, so a builder that sets more than that, chunking, filters, an extensible shape, an attribute, or a fill value, is rejected by the staging call. [`with_vlen_strings`](crate::DatasetBuilder::with_vlen_strings) is not one of them: overwriting a variable-length-string dataset places a fresh global heap collection and resolves the staged element references against it, on a contiguous, compact, or chunked (including filtered) dataset alike.

Overwriting the same dataset again reclaims the collection the previous overwrite placed, so rotating a dataset's strings within a session reaches a steady file size. Only this session's own placements are reclaimed, and only while nothing else can reach them: an in-file [`copy`](crate::File::copy) of a variable-length dataset re-emits its element references verbatim, so the copy refers to the same collections the source does. Any copy, or any raw-bytes write ([`with_raw_data`](crate::DatasetBuilder::with_raw_data)) over a datatype that reaches a heap address, gives up the reclaim for the rest of the session. Supplying raw bytes also drops whatever staging described the bytes they replace, so the two cannot be combined to slip past that. The collections a dataset held when the session *opened* are never reclaimed, since whatever wrote the file owns them, so [`repack`](crate::repack()) is what recovers those.

[`with_path_references`](crate::DatasetBuilder::with_path_references) is rejected, because its staged element bytes are placeholder addresses only a newly created dataset resolves. That is a rejection of the builder, not of the datatype it produces: a reference dataset is still overwritten by supplying bytes that need no resolving ([`with_reference_data`](crate::DatasetBuilder::with_reference_data), [`with_raw_data`](crate::DatasetBuilder::with_raw_data)). An object reference supplied as an address is screened at [`commit`](crate::File::commit) against the objects that commit deletes and the headers it rewrites elsewhere, the same ground a path target is checked on. The limitations reference has the screening rules.

## Choosing the fsync cadence

Those barriers are `fsync`s, and by default there is one at every durability point: five per append *batch* (a [`Dataset::append`](crate::Dataset::append) larger than about a megabyte is applied in several), two or three per commit, one at [`close`](crate::File::close). That is a strong guarantee and a real cost, and an application that wants durability at its own cadence instead sets [`SyncPolicy`](crate::SyncPolicy) on the fapl.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("log.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("samples")
#     .with_f64_data(&[])
#     .with_shape(&[0])
#     .with_maxshape(&[u64::MAX])
#     .with_chunks(&[256]);
# builder.write(&path)?;
use hdf5_pure::{File, FileAccessProperties, SyncPolicy};

let file = File::open_rw_with_options(
    &path,
    FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
)?;

let mut samples = file.dataset("samples")?;
for batch in 0..1000 {
    samples.append(&[f64::from(batch); 64])?;  // no fsync
}
file.close()?;   // applies staged edits, then one fsync
# drop(samples);
# assert_eq!(File::open(&path)?.dataset("samples")?.shape()?, vec![64_000]);
# Ok::<(), hdf5_pure::Error>(())
```

[`SyncPolicy::OnClose`](crate::SyncPolicy::OnClose) is close to what the reference C library does: its default `sec2` driver installs no flush callback at all, so `H5Fflush` drains libhdf5's caches with `write` and stops there, leaving power-loss durability to the application. It is not a write-back cache: every commit and every append has reached the operating system by the time it returns, under either policy. [Write gathering](#write-gathering) covers what is held back and for how long. A file written under [`OnClose`](crate::SyncPolicy::OnClose) is byte-identical to the same file written under [`Always`](crate::SyncPolicy::Always), is visible to other processes on the machine once the operation that wrote it returns, and survives *this process* crashing.

Three things are given up, and they are worth separating:

- **Power-loss ordering.** With the barriers gone, a machine that loses power mid-commit can have the superblock repoint on disk without the data it points at.
- **Deferred write errors.** On a filesystem that allocates late, a write that will fail at writeback still returns success, and the `fsync` is where the `ENOSPC` or `EIO` surfaces. Skip it and a commit the filesystem cannot complete returns `Ok`, with nothing left to report it.
- **Cross-host visibility.** "Another process sees it" is the page cache, so it holds on a local filesystem. Under NFS's close-to-open semantics a client may hold writes until a flush, so a reader on another host, a SWMR reader included, may not see them.

### Why the last one is not optional

[`File::sync()`](crate::File::sync) is the checkpoint the application issues itself, wherever it wants one. It writes nothing: staged edits still need a [`commit()`](crate::File::commit), and elements held by a [`BufferedAppender`](crate::BufferedAppender) need a [`flush()`](crate::BufferedAppender::flush). It only forces what is already written.

The barrier at the end is a different thing, and it is the reason this policy is [`OnClose`](crate::SyncPolicy::OnClose) and not a literal "never". [`close()`](crate::File::close) is an active call: it applies any staged edits, re-homes the free-space managers of a file that persists them, and clears a SWMR writer's flag, and it *consumes the handle*, so those writes land past the last point a caller could have ordered them. Dropping the last handle does the same, with no return value to report an error through. A policy that skipped the barrier there would take a cadence away, since no [`sync()`](crate::File::sync) you could write reaches those bytes. So [`close`](crate::File::close) and `drop` each issue exactly one `fsync`, under every policy.

The arithmetic makes this a cheap promise to keep: one barrier per session, against the five per append batch and two or three per commit that [`OnClose`](crate::SyncPolicy::OnClose) removes. A thousand-append session goes from about five thousand `fsync`s to one.

A file left flagged by a SWMR writer is the sharpest case. That flag is cleared at teardown, and a lost clear means every later open is rejected with [`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse) until [`File::clear_swmr_flag`](crate::File::clear_swmr_flag) runs, a failure that costs availability, on a write the caller never sees.

The whole-file paths are outside all of this: [`FileBuilder::write`](crate::FileBuilder::write) and [`repack`](crate::repack()) never `fsync` under either policy, since each writes a file and hands it over, with no editing session to hold.

The default is [`SyncPolicy::Always`](crate::SyncPolicy::Always), which is every guarantee described above this section.

## Write gathering

A commit or an in-place append is not one write. A single append writes the chunk's bytes at end-of-file and the superblock's recorded end-of-file, then patches the chunk index, the array header, and the dataspace dimension in the object header: five writes on a measured file, four of them a few dozen to a hundred-odd bytes, landing in a handful of pages that the *next* append patches again. Issued one at a time, that is four syscalls and four page dirtyings for a few hundred bytes of bookkeeping, and on flash a page dirtied four times is written four times. (An append that has to allocate a new extensible-array data block costs eight, which the first append into a dataset always does.)

Every read-write session therefore gathers its writes and emits **one write per dirty page**. The bytes it holds are released at every *ordering barrier*, the points the commit and append sequences already define, where writes made before must reach the disk before writes made after. Three things follow:

- A commit or an append that has returned has put its bytes in the operating system, whatever the [`SyncPolicy`](crate::SyncPolicy) says, because every operation ends with a barrier.
- Crash safety is unchanged: the barriers keep their ordering meaning under both policies, so a write that fails still leaves the file in the state it had before the operation.
- The resulting file is unchanged, byte for byte.

How much that saves depends on how much of an operation falls between two barriers. Measured on a paged file, a session appending one chunk to each of eight datasets, four times over, and then committing eight staged dataset creations: the appends issue all 184 of the write calls they make, while the commit's 24 go out as 4. An append's writes are separated by the barriers its own ordering needs, so there is almost nothing left to merge: four of the five land in two pages, two apiece, and all five still go out separately. A commit's tail puts many writes into a handful of pages as it rebuilds a group, repoints a root and re-homes the free-space managers, which is what merging within a barrier is for.

There is no setting for the gathering and nothing to opt into. Merging *across* barriers is what [the next section](#letting-pages-live-longer) adds. The one session that does *not* gather is the SWMR writer, whose readers follow its ordered phases as they become visible. Coalescing those would leave the file the same size, and would let a reader see a state the phases exist to hide.

### Letting pages live longer

Under the default, a page dirtied on either side of a barrier is written twice. `H5Pset_page_buffer_size`'s analogue lifts that:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("ring.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("t0").with_i32_data(&[0]);
# builder.write(&path)?;
use hdf5_pure::{File, FileAccessProperties, SyncPolicy};

let file = File::open_rw_with_options(
    &path,
    FileAccessProperties::new()
        .with_sync_policy(SyncPolicy::OnClose)
        .with_page_buffer_size(4 << 20),
)?;
# file.close()?;
# Ok::<(), hdf5_pure::Error>(())
```

Dirty pages then survive across barriers, commits and appends until the budget is spent, an `fsync` is issued, or the session closes. Measured on a paged file, 32 chunk appends into eight datasets followed by a commit: **188 writes with the default gathering, 5 with a page buffer**, two of which are the mark below going up and coming down. The appends issue nothing at all until the session ends. The file it produces is byte-identical either way.

Four things are required, and a page buffer that lacks one is rejected, never quietly ignored:

- **A budget of at least the page the session merges within**: the file's own page size when it was created with [`FileSpaceStrategy::Page`](crate::FileSpaceStrategy::Page), and the format's 4 KiB default otherwise. A buffer that cannot hold one page drains on every page it touches.
- **Persisted free space, on a paged file**, which without it can neither be committed to nor appended to, so the buffer would hold nothing while its mark blocked every reader. An unpaged file has no such constraint.
- **A version-3 superblock**, because the session marks the file while it holds it and nothing reads that byte back on an older one. See below.
- **[`SyncPolicy::OnClose`](crate::SyncPolicy::OnClose).** Under the default [`Always`](crate::SyncPolicy::Always) every barrier is an `fsync` that flushes the buffer, so it would hold nothing while still costing the mark.

There is also **a session long enough to pay for the mark**, which is a trade and not a rejection. See below.

[`File::create_with_options`](crate::File::create_with_options) rejects a creation/access pair it could not reopen with, before writing the file, and the SWMR writer rejects a page buffer outright: its readers observe the order its writes become visible in, which is exactly what a buffer coalesces away.

### Choosing a budget

Any budget of at least one page is accepted, and one below 1 MiB is a request for less resident memory: a writer inside a tight memory cap can set 256 KiB and get it. What it buys that memory with is writes: the budget is the point at which everything held is flushed, so a long contiguous run is flushed and restarted once per budget's worth of it. Writes issued, measured on a 4 KiB-paged file:

| workload | unset | 4 KiB | 64 KiB | 1 MiB |
| --- | --- | --- | --- | --- |
| 32 chunk appends into 8 datasets, then a commit | 188 | 25 | 4 | 4 |
| one 4 MiB append | 131 | 1,094 | 74 | 10 |

On the scattered workload a page buffer exists for, a small budget costs little. On the long run 64 KiB is seven times the writes of 1 MiB.

The memory comparison against leaving the property unset is not the one that table suggests. A session that sets nothing already gathers up to 1 MiB of dirty bytes **per operation** and releases it at every barrier. A page buffer holds its budget **across** operations, until the budget is spent, an `fsync`, or [`close`](crate::File::close). So 1 MiB here trades a per-operation peak for a continuous residency of the same size, and a budget below 1 MiB lowers both.

### Where this differs from `H5Pset_page_buffer_size`

**A paged file is not required.** `H5PB_create` requires one, because the C page buffer is a page *cache* whose `min_meta_perc` / `min_raw_perc` reservations count pages the paged allocator keeps segregated by kind. This is a write gatherer instead: it merges runs within a page-sized window and flushes whole, so a window is all it needs, and an unpaged file gets the same 4 KiB one every read-write session already gathers under. Unpaged is the default strategy, so requiring [`Page`](crate::FileSpaceStrategy::Page) would put the property out of reach of most files, and this implementation has no need of it.

**A small budget costs writes here, where it costs none in C**, because `H5PB_write` bypasses the C buffer for any I/O of a page or more, so capping memory there does not throttle a long write. Nothing bypasses this gatherer, so a small budget is also where a long run is flushed and restarted. Both libraries accept the budget, and only this one charges for it.

**A sub-page budget is rejected, never rounded.** `H5Fcreate` rejects it too, `H5Fopen` rounds it up to one page silently, and on an unpaged file `H5Fopen` silently zeroes the budget outright. A property quietly ignored is worse than one rejected.

### What a page buffer trades, and what pays for it

Gathered writes go out in address order, and every publish point sits below the content it reaches: a commit's superblock and root at address 0, an append's dataspace dimension and chunk-index element count in the object header near the front, the chunk bytes and index blocks those point at, at end-of-file. Held across their barriers, all of them are issued first. A write that fails, or a process that dies, mid-flush can therefore leave:

- a root or an end-of-file pointing at bytes that never arrived, a file that fails to read, which is the benign case
- a dataset whose length was published but whose rows were not, which reads back **clean**, as fill values
- a dataset header published over a region an earlier commit freed, which reads back **clean**, as the deleted object's data

The last two are silent: every checksum verifies and a reader has no signal. This is what a write-back page buffer *is*, and `H5Pset_page_buffer_size` reorders the same way, with the same silence on crash consistency. The guarantee being switched off is one this crate adds on top of HDF5, and not one an HDF5 user arrives expecting.

So a page-buffered session **marks the file for its lifetime**: superblock status-flag bit 0, `H5F_SUPER_WRITE_ACCESS`, the byte the reference C library raises for any writer and this crate otherwise raises only for a SWMR one. It is written and `fsync`ed at open, republished by every commit, and cleared on a clean [`close`](crate::File::close) or drop. A session that dies with pages in memory leaves it standing, and a file carrying it is rejected by this crate, by `H5Fopen` and by h5py alike, with [`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse). The silent wrong result becomes an error. [`File::clear_swmr_flag`](crate::File::clear_swmr_flag) (the `h5clear -s` equivalent) is how to look at such a file anyway, knowing what it may hold.

A reader has a snapshot while the session is still running. The mark is bit 0 alone, which no SWMR reader can follow, so [`FileAccessProperties::with_write_mark_policy(WriteMarkPolicy::AllowSnapshot)`](crate::FileAccessProperties::with_write_mark_policy) reads the file as it stands through any `*_with_options` open, on the caller's assertion that the writer has flushed, with [`File::sync`](crate::File::sync) or a clean close. It admits nothing else: a SWMR pair still belongs to [`File::open_swmr`](crate::File::open_swmr), and no reader's opt-in lets a second writer in.

### When it is worth it

The mark costs two `fsync`s per session, at open and at close, whatever the session then does, so a page buffer pays off over a long session and costs on a short one. Measured on an Apple M1 Max (APFS), 256-byte appends into eight datasets, arms alternated and medians of seven:

| appends | default | page buffer | ratio |
|---:|---:|---:|---:|
| 400 | 14.5 ms | 19.3 ms | 0.75 |
| 800 | 23.3 ms | 24.3 ms | 0.96 |
| 1,600 | 41.0 ms | 33.3 ms | 1.23 |
| 3,200 | 79.1 ms | 54.8 ms | 1.45 |
| 6,400 | 152.9 ms | 93.5 ms | 1.64 |

Break-even is near 800 appends here, and the ratio climbs past it because the same pages are re-dirtied more often the longer the session runs. The 1,600-append session ran 1.64 with the mark removed against 1.23 with it, which is what the guarantee costs, a fixed price per session and not a rate. The margin narrows to about 1.1 once 64 KiB payloads dominate over metadata churn.

These are one host's numbers and the short end is noisy, since both of its arms are small enough for the fixed `fsync` cost to dominate. `cargo bench --bench hot_paths -- page_buffer` runs both sides of the crossing on yours.

## Supported targets and formats

Contiguous and chunked datasets (with any filter the whole-file writer supports) and compact-link groups are supported. The editor works across every on-disk format the reference HDF5 C library and h5py produce:

- Version 0, 1, 2, and 3 superblocks.
- Single- and multi-chunk object headers. A multi-chunk header is collapsed into a single chunk on rewrite.
- A version 0/1 symbol-table group on the edited path is converted to the latest compact-link format. Adding and deleting are supported on these older files, and copying a version-1 object is not.

The editor rejects anything it cannot reproduce faithfully, with [`Error::EditUnsupported`](crate::Error::EditUnsupported), and never degrades a file silently:

- A file whose superblock is not located at its base address, a relocated or malformed userblock layout. (A canonical userblock, such as a MATLAB v7.3 `.mat` file's 512-byte userblock, is supported: addresses are read and written relative to the base and the userblock bytes are preserved.)
- Dense-storage headers on the edited path.
- Editing a **group with more than one hard link**, or one whose link count cannot be established because the file's links cannot be walked. A commit rebuilds a dirty group's object header at a fresh address and repoints the link it resolved the group through, and every other link would be left pointing at the old header, which the same commit frees. The rejection reaches an edit *below* such a group too, since every ancestor on the edited path is rebuilt. The same rule governs a relocating dataset write, append, or attribute edit. The root group is exempt, since the superblock addresses it and no link does. Deleting a link is unaffected, since it relocates nothing.
- Copying an existing version-1 object.
- Across files ([`copy_from`](crate::File::copy_from)): variable-length or reference datasets and attributes, any shared (committed/SOHM) header message, and a streaming source file, none of which can be reproduced verbatim in another file.
- Editing the attributes of an object whose attribute message is a **shared (SOHM) record**, and moving or removing an object header the file's shared-message index lists. A file that shares messages is read and edited otherwise. What a commit cannot do is change the table's reference count or repoint the index, so it rejects the two edits that would need to. [`repack`](crate::repack()) rewrites such a file with every message stored inline ([#417](https://github.com/CramBL/hdf5-pure/issues/417)).
- Copying a dataset whose elements live in **external files** (`H5Pset_external`), in either direction. It carries the same address-less contiguous layout a never-written dataset does, which is copied as the empty storage it is, so the External Data Files message tells the two apart, where the address cannot ([#336](https://github.com/CramBL/hdf5-pure/issues/336)).

[`Error::EditUnsupported`](crate::Error::EditUnsupported) carries the reason as its payload.

## Object references the file already stores

An object reference is an object-header address stored as data, in a dataset's elements or in an attribute's value. A commit rebuilds every dirty object's header at a fresh address and repoints the *link* that resolves to it, which leaves those stored addresses pointing at a header the same commit frees. So a commit's last act, after the superblock repoint that publishes it, is to walk the tree it just published and rewrite every stored address that points at something it moved. Nothing about this is opt-in and no API reports it, and the point is that a reference keeps reaching its object.

Two things decide what it reaches. **Where the bytes are**: a contiguous or compact dataset's elements, and an attribute held in the object header, and none of a chunked dataset's chunks, a dense (fractal-heap) attribute, or one held as a shared (SOHM) record. A version 1 object header is read by a different parser that reaches element data and stops short of an attribute's value. Version 1 is the format the reference C library and h5py write by default, so this is the common case, and not a legacy corner. And **what the datatype is**: an 8-byte object reference, an array of them, a compound holding one, and any nesting of those two, and none of an object reference wider than 8 bytes, a dataset-region reference, a variable-length reference (how the dimension-scale attribute `DIMENSION_LIST` is encoded), or an enumeration over one.

Everything outside that is left exactly as it was, since rejecting it would ban editing such a file at all, because every commit rebuilds its root group ([#324](https://github.com/CramBL/hdf5-pure/issues/324)).

That argument covers what a file already is, and not what an edit makes it. An attribute edit that would move an attribute holding a *repointable* reference out of the object header, where this walk reaches it, into a heap, where it would not, is **rejected** ([#102](https://github.com/CramBL/hdf5-pure/issues/102)): the reference would otherwise stop being repointable for good, since an object that goes dense stays dense. The rejection costs one edit, and the file stays editable. An attribute whose references this walk never repointed anyway, a `DIMENSION_LIST`, a region reference, one wider than 8 bytes, moves freely, and an object the file already stored densely is rebuilt as it is.

A reference whose target the commit **deletes** is left alone too, and for a different reason: the object is gone, and HDF5 has no rule that turns a reference to it into something else. The reference C library behaves the same way.

The cost is one pass over the file's objects per commit, so it scales with the object count and not with the file's size. A session that proves a file is free of references skips the pass on every later commit, until it stages something that could introduce one.

## Space reuse and truncation

Within a session, the space a deletion frees is reused for later writes, so add/delete churn keeps the file bounded. If a freed run reaches the end of the file, the file is truncated to where that run starts, so [`File::file_size()`](crate::File::file_size) reports the shorter length and not a high-water mark. On a paged file the end of allocation stays a whole number of pages, so what comes back is whole pages.

Two things stay behind. A file that persists its free space rewrites its free-space managers on every commit and cannot put them where the previous set still is, so a few of those blocks' worth of the run is kept for the commits that follow. A run that would give back less than it keeps is left where it is, since trimming the top off a hole the next write was about to fill costs more than it returns. The result is that the file ends just above its last live allocation, and a delete can still make a file longer: recording free space on disk costs metadata that has to go somewhere, and the reference C library grows a file to persist its managers too.

Everything a commit writes can go into freed space: object headers, contiguous data, a chunked dataset's chunk data and index, and a dense attribute heap. Each needs a region large enough to hold it whole, so a dataset larger than every free region is written past the end of the file, never split across two holes.

Contiguous and chunked datasets (chunk index plus chunk data) and whole group subtrees are reclaimed. Reclaim is best-effort: an object whose blocks cannot be enumerated exhaustively (variable-length global-heap storage, dense attribute or link heaps, a version 2 B-tree chunk index) is left as dead bytes, which never risks freeing a region still in use.

On a paged file ([`FileSpaceStrategy::Page`](crate::FileSpaceStrategy::Page)) an allocation is served from free space of the page type it is writing, or from a page that is wholly free, since a page with nothing in it has no type to contradict, so reuse cannot make metadata and raw data share a page. Freed space whose page type cannot be established, a whole-file-generic free section recorded by another writer, or a chunk index that writer placed among its metadata, is recorded and never handed out, which costs some space and keeps the page separation.

On any file that persists its free space, the blocks a commit rewrites to record it, the superblock extension and the free-space managers themselves, are placed in free space where any fits, and appended only when none does. Each commit supersedes the previous set, so in the steady state a commit's blocks land in space an earlier commit's vacated, and a delete-and-recreate workload holds the file at its size, where each commit would otherwise add a set (paged: a page). When a commit had to append those blocks *above* the run it just freed, the shape a delete leaves, it moves them down into that run in a second, tail-only rewrite and truncates to them, which is how the delete-then-commit case gives its space back within the same session.

An immediate [`Dataset::append`](crate::Dataset::append) (and a [`BufferedAppender`](crate::BufferedAppender) flush) draws on free space too. Which space depends on how the file records it. A file that forgets its free space at close spends it directly: nothing on disk claims those bytes are free, so a crash mid-append leaves them as dead as they were. A file that **persists** its free space, every paged file included, cannot do that, because an append has no superblock repoint to update the on-disk managers with. Such a session instead takes a batch of space *out* of those managers first, in a rewrite of its own, and spends only from there, so no byte an append writes is ever one the managers advertise, at any instant, and a crash can strand the unspent remainder of a batch and can never hand it out twice. A batch gathers every hole the appended chunk fits in, up to a megabyte of them, so one rewrite covers many holes and the file grows only when no hole can hold the chunk. The SWMR writer always appends at the end of the file: its readers may still be inside a region the session freed. One consequence of reuse matters for read-heavy workloads: a dataset written into a fragmented file may have its chunks placed in several holes, and a sequential read of it then fetches each chunk separately, where a single run coalesces adjacent ones. [`repack`](crate::repack()) restores a single run.

By default, freed space is reused only within the open session and forgotten on close. For a file created with `H5Pset_file_space_strategy(persist = true)`, freed space is recorded on disk and survives a reopen, which the [file-space strategy](crate::_guide::file_space) guide sets out. For a guaranteed shrink that rewrites the whole file compact across a reopen, use [`repack`](crate::repack()), and the [repack](crate::_guide::repack) guide has the options.

## Verifying edits

Reopen the file with [`File::open`](crate::File::open) to confirm the edits landed:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("temperature").with_f64_data(&[22.5, 23.1, 21.8]);
# let mut sensors = builder.create_group("sensors");
# sensors.create_dataset("pressure").with_f32_data(&[101.3]);
# builder.add_group(sensors.finish());
# builder.write(&path)?;
# let editing = File::open_rw(&path)?;
# let editing_root = editing.root();
# editing_root.create_group_with("run2", |g| {
#     g.set_attr("kind", AttrValue::AsciiString("trial".into()));
# })?;
# editing_root.create_dataset("run2/signal", |b| {
#     b.with_f64_data(&[1.0, 2.0, 3.0]);
# })?;
# editing.copy("temperature", "temperature_backup")?;
# editing_root.delete("sensors/pressure")?;
# editing.commit()?;
# drop(editing_root);
# drop(editing);
use hdf5_pure::{AttrValue, Error, File, FormatError};

let file = File::open(&path)?;
let signal = file.dataset("run2/signal")?.read_f64()?;
let backup = file.dataset("temperature_backup")?.read_f64()?;
let run2_attrs = file.group("run2")?.attrs()?;

assert_eq!(signal, vec![1.0, 2.0, 3.0]);
assert_eq!(backup, vec![22.5, 23.1, 21.8]);
assert_eq!(run2_attrs["kind"], AttrValue::AsciiString("trial".into()));

let err = file.dataset("sensors/pressure").unwrap_err();
let Error::Format(FormatError::PathNotFound(missing)) = &err else {
    panic!("expected PathNotFound, got {err:?}");
};
assert_eq!(missing, "pressure");
# Ok::<(), hdf5_pure::Error>(())
```

For background on the append-and-repoint design, see the [crate documentation](crate).
